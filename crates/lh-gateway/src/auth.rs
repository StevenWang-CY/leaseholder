//! Tenants and API-key authentication.

use std::sync::Arc;

use hmac::{Hmac, KeyInit, Mac};
use http::HeaderMap;
use http::header::AUTHORIZATION;
use lh_proto::SloClass;
use sha2::{Digest, Sha256};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::config::{self, CacheSaltSecret};
use crate::sanitize::hex;

/// A tenant as the request path sees it.
#[derive(Debug)]
pub(crate) struct Tenant {
    pub name: Arc<str>,
    pub trusted: bool,
    pub default_class: SloClass,
    pub max_class: SloClass,
    /// The `cache_salt` set on every engine request of this tenant: its prefix-cache namespace.
    pub cache_salt: String,
}

#[derive(Debug)]
pub(crate) struct Tenants {
    tenants: Vec<Tenant>,
    /// Every key digest with the index of its tenant.
    keys: Vec<([u8; 32], u32)>,
}

impl Tenants {
    pub fn new(config: &[config::Tenant], secret: &CacheSaltSecret) -> Self {
        let tenants = config
            .iter()
            .map(|t| Tenant {
                name: t.name.as_str().into(),
                trusted: t.trusted,
                default_class: t.default_class,
                max_class: t.max_class,
                cache_salt: cache_salt(secret, &t.name),
            })
            .collect();
        let keys = config
            .iter()
            .zip(0u32..)
            .flat_map(|(t, index)| t.keys_sha256.iter().map(move |digest| (*digest, index)))
            .collect();
        Self { tenants, keys }
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.tenants.iter().position(|t| &*t.name == name)
    }

    /// The tenant of a request: the owner of its bearer key, else the listener's tenant. On a listener without
    /// one, a missing or unknown key yields `None` (401).
    pub fn authenticate(
        &self,
        headers: &HeaderMap,
        listener_tenant: Option<usize>,
    ) -> Option<&Tenant> {
        let index = bearer_token(headers)
            .and_then(|token| self.match_key(token))
            .or(listener_tenant)?;
        self.tenants.get(index)
    }

    /// The tenant whose key digest equals SHA-256 of `token`. Every digest is compared in constant time and the
    /// loop never exits early, so timing reveals neither which key matched nor how much of a digest did.
    fn match_key(&self, token: &[u8]) -> Option<usize> {
        let digest: [u8; 32] = Sha256::digest(token).into();
        let mut found = Choice::from(0);
        let mut index = 0u32;
        for (key, tenant) in &self.keys {
            let equal = key[..].ct_eq(&digest[..]);
            index.conditional_assign(tenant, equal);
            found |= equal;
        }
        bool::from(found).then_some(index as usize)
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&[u8]> {
    let (scheme, token) = headers.get(AUTHORIZATION)?.as_bytes().split_at_checked(7)?;
    scheme.eq_ignore_ascii_case(b"bearer ").then_some(token)
}

/// The first 32 hex digits of HMAC-SHA-256 of the tenant's name, keyed by the configuration's secret. It depends
/// only on the tenant, so rotating a key keeps the tenant's cache, and a keyless tenant gets one too; without the
/// secret, the name does not give it away.
pub(crate) fn cache_salt(secret: &CacheSaltSecret, tenant: &str) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(&secret.0).expect("HMAC takes a key of any length");
    mac.update(tenant.as_bytes());
    hex(&mac.finalize().into_bytes()[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    /// The secret of the bytes `first` to `first + 31`.
    fn secret(first: u8) -> CacheSaltSecret {
        CacheSaltSecret(std::array::from_fn(|i| first + i as u8))
    }

    fn tenants() -> Tenants {
        let tenant = |name: &str, keys: &[&str]| config::Tenant {
            name: name.into(),
            keys_sha256: keys.iter().map(|k| Sha256::digest(k).into()).collect(),
            trusted: false,
            default_class: SloClass::Interactive,
            max_class: SloClass::Interactive,
        };
        Tenants::new(
            &[
                tenant("voice", &[]),
                tenant("agents", &["agents-1", "agents-2"]),
                tenant("bench", &["bench-1"]),
            ],
            &secret(0),
        )
    }

    fn bearer(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_str(value).unwrap());
        headers
    }

    fn tenant_of(
        tenants: &Tenants,
        headers: &HeaderMap,
        listener: Option<usize>,
    ) -> Option<String> {
        tenants
            .authenticate(headers, listener)
            .map(|t| t.name.to_string())
    }

    #[test]
    fn every_key_position_matches_its_tenant() {
        let tenants = tenants();
        for (key, owner) in [
            ("agents-1", "agents"),
            ("agents-2", "agents"),
            ("bench-1", "bench"),
        ] {
            let headers = bearer(&format!("Bearer {key}"));
            assert_eq!(tenant_of(&tenants, &headers, None).as_deref(), Some(owner));
        }
        assert_eq!(
            tenant_of(&tenants, &bearer("bearer bench-1"), None).as_deref(),
            Some("bench")
        );
    }

    #[test]
    fn unknown_keys_fall_back_to_the_listener_tenant_or_fail() {
        let tenants = tenants();
        let voice = tenants.index("voice");
        for headers in [
            HeaderMap::new(),
            bearer("Bearer bench-2"),
            bearer("Bearer bench-1 "),
            bearer("Basic bench-1"),
            bearer("Bearer"),
        ] {
            assert_eq!(tenant_of(&tenants, &headers, None), None, "{headers:?}");
            assert_eq!(
                tenant_of(&tenants, &headers, voice).as_deref(),
                Some("voice")
            );
        }
        // A known key wins over the listener's tenant.
        assert_eq!(
            tenant_of(&tenants, &bearer("Bearer agents-2"), voice).as_deref(),
            Some("agents")
        );
    }

    #[test]
    fn cache_salts_are_pinned_per_tenant_and_secret() {
        // `printf agents | openssl dgst -sha256 -mac HMAC -macopt hexkey:000102...1f`, its first 32 hex digits: a
        // change would orphan every tenant's prefix cache, so the values are pinned.
        assert_eq!(
            cache_salt(&secret(0), "agents"),
            "3c112a3140ece6fbba30769936c5f17f"
        );
        assert_eq!(
            cache_salt(&secret(0), "voice"),
            "b9d705094e95b446146e3c5671bc3c6d"
        );
        // Another secret, other salts: without the secret, a tenant's name does not give its salt away.
        assert_eq!(
            cache_salt(&secret(1), "agents"),
            "34f06bd8e6338e453d4466724a323b0d"
        );
        assert_ne!(
            cache_salt(&secret(1), "voice"),
            cache_salt(&secret(0), "voice")
        );
    }
}

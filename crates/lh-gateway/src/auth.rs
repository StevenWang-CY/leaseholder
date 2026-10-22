//! Tenants and API-key authentication.

use std::sync::Arc;

use http::HeaderMap;
use http::header::AUTHORIZATION;
use lh_proto::SloClass;
use sha2::{Digest, Sha256};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::config;
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
    pub fn new(config: &[config::Tenant]) -> Self {
        let tenants = config
            .iter()
            .map(|t| Tenant {
                name: t.name.as_str().into(),
                trusted: t.trusted,
                default_class: t.default_class,
                max_class: t.max_class,
                cache_salt: cache_salt(&t.name),
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

/// The first 32 hex digits of SHA-256("leaseholder-cache-salt", NUL, tenant). It depends only on the tenant, so
/// rotating a key keeps the tenant's cache, and a keyless tenant gets one too.
pub(crate) fn cache_salt(tenant: &str) -> String {
    let digest = Sha256::new()
        .chain_update(b"leaseholder-cache-salt\0")
        .chain_update(tenant)
        .finalize();
    hex(&digest[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    fn tenants() -> Tenants {
        let tenant = |name: &str, keys: &[&str]| config::Tenant {
            name: name.into(),
            keys_sha256: keys.iter().map(|k| Sha256::digest(k).into()).collect(),
            trusted: false,
            default_class: SloClass::Interactive,
            max_class: SloClass::Interactive,
        };
        Tenants::new(&[
            tenant("voice", &[]),
            tenant("agents", &["agents-1", "agents-2"]),
            tenant("bench", &["bench-1"]),
        ])
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
    fn cache_salts_are_pinned_per_tenant() {
        // `printf 'leaseholder-cache-salt\0agents' | shasum -a 256 | cut -c1-32`: a change would orphan every
        // tenant's prefix cache, so the values are pinned.
        assert_eq!(cache_salt("agents"), "5aba4e96998029ea96716cc693e1703e");
        assert_eq!(cache_salt("voice"), "32e853135c92a8afaf0bf4b3a2afd65c");
    }
}

#!/usr/bin/env bash
# deploy/phastform/variant.sh on a GPU node in the campaign.sh tests: logs what it was asked to run.
echo "variant.sh $*" >>"$HOME/variant.log"

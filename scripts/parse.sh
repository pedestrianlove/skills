#! /usr/bin/env bash

REPO_NAME=$1

mkdir -p skills/$REPO_NAME

yq -n ".name = \"$REPO_NAME\" | .description = \"Skills for agents to consume for $REPO_NAME\"" --yaml-output --explicit-start > skills/$REPO_NAME/SKILL.md
echo "---" >> skills/$REPO_NAME/SKILL.md
rg -N --heading -o -U --no-ignore --replace '$1' '(?m)^(def\s+[a-zA-Z0-9_]+\s*\([\s\S]*?\)\s*:)' --type py third_party/$REPO_NAME >> skills/$REPO_NAME/SKILL.md

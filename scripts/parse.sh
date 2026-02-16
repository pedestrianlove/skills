#! /usr/bin/env bash

REPO_NAME=$1

mkdir -p skills/$REPO_NAME

pixi run rg -N --heading -o -U --no-ignore --replace '$1' '(?m)^(def\s+[a-zA-Z0-9_]+\s*\([\s\S]*?\)\s*:)' --type py third_party/$REPO_NAME > skills/$REPO_NAME/SKILL.md

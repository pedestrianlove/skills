#! /usr/bin/env bash

parallel --colsep ' ' bash scripts/clone-or-pull.sh {1} "third_party/{1/.}" :::: list.md

parallel --colsep ' ' bash scripts/parse.sh {1/.} {2} :::: list.md

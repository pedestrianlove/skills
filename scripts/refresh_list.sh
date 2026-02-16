#! /usr/bin/env bash

parallel bash scripts/clone-or-pull.sh {} "third_party/{/.}" :::: list.md

parallel bash scripts/parse.sh {/.} :::: list.md

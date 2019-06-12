#!/usr/bin/env bash
git pull origin master
git add -A
git commit -a -m "colab add"
git push -f origin master

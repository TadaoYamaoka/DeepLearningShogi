@echo off
call activate DeepLearningShogi
python -m dlshogi.usi.usi_policy_player 2>NUL

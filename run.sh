#DECOMPOSED_DIR = ./dataset/decomposed
#NBC_DIR = ./dataset/NBC
#TEST_DIR = ./dataset/selfcheckgpt
#WEB_DIR = ./dataset/webpage

C_M = 28
C_FA = 96
#C_RETRIEVE = 1

echo $C_M $C_FA
python -m main --C_M $C_M --C_FA $C_FA
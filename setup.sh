#!/bin/bash

echo "Configurando o ambiente virtual"

if [ ! -d "venv" ]; then
    python -m venv venv
fi

echo "Ativando o ambiente virtual"
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate # Windows (Git Bash)
else
    source venv/bin/activate     # Linux/Mac
fi

echo "Instalando dependências"
python -m pip install --upgrade pip

echo "Instalando pacotes necessários"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install pandas numpy opencv-python matplotlib openpyxl ipykernel
    pip freeze > requirements.txt
fi

echo "Ambiente virtual configurado com sucesso!"
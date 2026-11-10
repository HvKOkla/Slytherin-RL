
source venv/bin/activate
echo -e "\e[32mEnvironnement virtuel activé ! Prêt à coder.\e[0m"


run-tests() { 
    pytest tests/ -v
}

train-snake() { 
    python -m games.snake.training.train
}
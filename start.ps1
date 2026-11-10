.\venv\Scripts\activate
Write-Host "Environnement virtuel activé ! Prêt à coder." -ForegroundColor Green

function run-tests { 
    pytest tests/ -v 
}

function train-snake { 
    python -m games.snake.training.train 
}
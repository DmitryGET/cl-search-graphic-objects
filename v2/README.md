Чтобы запустить проект нужно: 
1. Скачать данную папку
2. Открыть консоль и прописать: ```docker build -t dpo_logo .```
3. После того как Docker образ успешно собрался, запускаем Docker-контейнер следующей командой: ```docker run -p 8000:8000 --name dpo_logo dpo_logo```

Поздравляем! Теперь можно загружать файлы и обнаруживать на них логотипы.

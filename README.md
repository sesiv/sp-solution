# sp-solution MVP

Этот репозиторий реализует Solution Architecture Contract (SAC.md) как минимальный, расширяемый MVP.

## Что реализовано
- FastAPI-транспорт с WebSocket-чатом (`/ws/{session_id}`) и SSE-событиями (`/events/{session_id}`).
- Менеджер сессий с очередями сообщений и брокером событий на сессию (backpressure с выбрасыванием старых).
- MCP-клиент браузера с поддержкой HTTP+SSE или WebSocket, handshake и восстановлением сессии.
- Конвейер наблюдений: парсит MCP-снимки (YAML или текст), удаляет дубликаты, ограничивает размер, детектит оверлеи.
- Политика безопасности: авторазрешение безопасных действий и запрос подтверждения на разрушительные клики (EN/RU).
- Агентный цикл LangGraph: plan -> observe -> act с валидацией действий, interrupt-потоками и лимитом шагов.
- Опциональный OpenRouter LLM для планов/действий/обновления состояния со structured output + скриншотами.
- CLI REPL (Typer + prompt_toolkit + Rich) с живыми SSE-логами и WebSocket-чатом/подтверждениями.

## Примечания по реализации
### Агентный цикл
- Режим чата использует LLM для plan/action/state; режим команд выполняет одиночные tool-команды.
- Лимит инструментальных шагов: 100; после 3 ошибок валидации действия блокируются.
- Валидация действий проверяет существование цели и допускает `type` только для редактируемых ролей.

### Наблюдение
- Парсит YAML-снимок MCP при наличии; иначе использует эвристику по тексту.
- Ограничивает интерактивные элементы (160) и текстовые блоки (30), обрезает блоки до 200 символов и дедупит.
- Детектит оверлеи (cookie/privacy/subscribe/и т.д.) и предлагает кандидаты для dismiss/accept.

### Политика и безопасность
- Allowlist: observe/launch/scroll/wait/screenshot/stop/need_user.
- Кликом с разрушительными ключевыми словами требуется подтверждение; остальные вне allowlist тоже требуют подтверждения.

### Транспорт и события
- WebSocket типы сообщений: `user_message`, `user_confirm`, `control`.
- Server message типы: `agent_message`, `agent_question`, `status`.
- SSE типы событий: `tool_call`, `tool_result`, `observation`, `policy_request`, `policy_result`, `error`.

### MCP интеграция
- Поддерживаются HTTP+SSE и WS endpoints; `MCP_ENDPOINT` обязателен.
- Маппинг инструментов на Playwright MCP defaults (`browser_snapshot`, `browser_click`, `browser_type`,
  `browser_press_key`, `browser_wait_for`, `browser_take_screenshot`, `browser_install`).
- Скролл адаптируется к `press_key`/`scroll`/`wheel` для совместимости.

## Быстрый старт
1) Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2) Запустите сервер:

```bash
sp-server
```

3) Запустите CLI (используйте `--auto-launch`, если хотите отправить `/launch` при старте):

```bash
sp-cli --server http://127.0.0.1:8000 --session demo
```

Затем один раз выполните `/launch`, если MCP-серверу нужен этап установки/инициализации браузера.

## Docker Compose
Сборка и запуск API + MCP сервера:

```bash
docker compose up --build
```

Примечания:
- Чтобы увидеть headed браузер на Linux, разрешите X11 и передайте `DISPLAY`:
  `xhost +local:docker` перед `docker compose up`.
- MCP скачивает пакеты при первом запуске через `npx`, поэтому нужен доступ в сеть.
- Если MCP не стартует, задайте `MCP_PACKAGE` или `MCP_CMD` в `.env` для переопределения команды.
- Для persistent-сессий в Docker задайте `MCP_ARGS` с `--user-data-dir=/data` (по умолчанию в `docker-compose.yml`).
- Chrome в Docker работает от root; `--no-sandbox` включен в `MCP_ARGS` по умолчанию.

## CLI на хосте (рекомендуется при нестабильном TTY в Docker)
Поднимите сервисы Docker, затем запустите CLI на хосте с локальным TTY:

```bash
./scripts/repl.sh
```

Скрипт запускает `docker compose up -d` и затем выполняет `sp-cli` через `workon sp_solution`.
Переопределяйте дефолты через `SP_SERVER` и `SP_SESSION`.

## Команды CLI
- `/chat` — вход в режим чата (agent loop).
- `/exit` — выход из режима чата и возврат в режим команд.
- `/quit` — выход из REPL.
- `/launch` — запуск инструмента установки/инициализации браузера MCP.
- `/observe` — новый цикл наблюдения.
- `/click <eid>` — клик по элементу.
- `/type <eid> <text>` — ввод текста в поле.
- `/scroll [direction] [amount]` — скролл (по умолчанию: down 600).
- `/wait [ms]` — ожидание.
- `/screenshot` — запрос скриншота.
- `/yes` или `/no` — подтверждение policy-действий.

Используйте `--auto-launch` или `SP_AUTO_LAUNCH=1` для автоматической отправки `/launch`.

## Конфигурация MCP
Браузерный слой ожидает endpoint MCP сервера Playwright. Настройте через переменные окружения:

- `MCP_ENDPOINT`: HTTP(S) или WS(S) endpoint MCP (например `http://127.0.0.1:3333/mcp`).
- Имена инструментов соответствуют Playwright MCP defaults (`browser_snapshot`, `browser_click`, `browser_type`,
  `browser_press_key`, `browser_wait_for`, `browser_take_screenshot`, `browser_install`).
- `MCP_USER_DATA_DIR`: каталог профиля браузера для persistent-сессий (используется MCP сервером).

Если `MCP_ENDPOINT` не задан, браузерный слой вернет явную runtime-ошибку.

## OpenRouter (опционально)
Задайте:
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (например `openai/gpt-4o-mini`)
- `OPENROUTER_PROVIDER` (опционально, порядок провайдеров через запятую)

Режим чата вызывает LLM только при наличии конфигурации; режим команд LLM не требует.

## Notes
- Наблюдение ограничивает объем данных и никогда не пересылает полный снимок страницы.
- Политика безопасности требует подтверждения для действий вне allowlist.
- Агент имеет лимит 100 инструментальных шагов и останавливается после повторных ошибок валидации.

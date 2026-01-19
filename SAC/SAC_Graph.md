# Solution Architect Contract  
## Переписывание AgentRunner на LangGraph

## 1. Цель
Переписать текущую реализацию агента с императивного `while`-loop на **LangGraph StateGraph**:
- явные ноды и переходы
- единый state вместо `RunState` и `Session.pending_*`
- human-in-the-loop через `interrupt / Command(resume=...)`
- без ручной оркестрации цикла в Python

Агент локальный, без persistence, рестарт сервера завершает выполнение.

---

## 2. Ключевые решения

- **Session становится тонким транспортным слоем**
  - очереди сообщений
  - EventBroker
  - status
  - session_id
- Источником истины является **state LangGraph**
- Chat history хранится **в state графа**
- Pending (confirm/manual) реализуется **только через interrupt**
- Один pending одновременно
- `MAX_TOOL_STEPS` считается **по всем нодам графа**
- Planner переводится на **structured output**, ручной JSON парсинг удаляется
- **Скриншот всегда берется при выполнении observe**
  - observe ⇒ screenshot + observe
  - если observe не выполняется, screenshot не делается

---

## 3. Что остается без логических изменений

- `models.py`
  - Action
  - Observation
  - Event
  - ServerMessage
  - PolicyDecision
- `policy.py`
  - PolicyGate.assess
- MCP слой
  - MCPBrowserClient
  - observe / click / type / scroll / wait / screenshot / launch

---

## 4. Что переписывается

### 4.1 AgentRunner
- Убирается `while`-loop
- Runner:
  - принимает ввод пользователя
  - вызывает graph `ainvoke`
  - прокидывает `interrupt` в UI
  - резюмирует graph через `Command(resume=...)`
- Runner не хранит состояние агента

### 4.2 session.py
- Session больше не хранит:
  - chat_history
  - pending_action / pending_kind
  - RunState
- Session используется только как транспорт и UI-адаптер

### 4.3 agent/llm.py
- Убирается ручной JSON parsing
- Planner использует structured output (pydantic schema)
- Возвращает типизированное действие без regex и `_extract_json`

### 4.4 Новый модуль графа
- `agent/langgraph_agent.py`
- Содержит:
  - State schema
  - StateGraph
  - все ноды
  - routing функции
  - checkpointer

---

## 5. Checkpointer
- Используется InMemorySaver
- `thread_id = session.session_id`
- Persistence между рестартами не требуется

---

## 6. State Schema

Минимальный state:

- `messages`: list[BaseMessage]
- `last_user_message`: str | None
- `last_observation`: Observation | None
- `last_screenshot`: dict[str, str] | None
- `planned_action`: Action | None
- `steps`: list[str]
- `failures`: list[str]
- `tool_steps`: int
- `invalid_actions`: int
- `needs_observe`: bool
- `final_response`: str | None

Chat history существует только в `messages`.

---

## 7. Ноды графа

### 7.1 ingest_user_input
- Обрабатывает:
  - `{user_message: str}`
  - `Command(resume=payload)`
- Добавляет HumanMessage
- Обновляет `last_user_message`

### 7.2 maybe_observe
- Если `needs_observe == True` или `last_observation is None`
  - **screenshot**
  - **observe**
- Скриншот и observe всегда выполняются вместе
- Обновляет:
  - `last_observation`
  - `last_screenshot`
  - `needs_observe = False`
- Публикует события observation

### 7.3 plan_action
- Формирует ActionContext из state
- Вызывает LLM planner
- Записывает `planned_action`

### 7.4 validate_action
- Проверяет:
  - существование eid
  - disabled
  - editable role для type
- При ошибке:
  - `invalid_actions += 1`
  - запись в failures и steps
- Routing:
  - `invalid_actions >= 3` → finalize_blocked
  - иначе → plan_action

### 7.5 policy_gate
- Вызывает `PolicyGate.assess`
- Routing:
  - allow → execute_action
  - requires_confirmation → confirm_interrupt
  - deny → finalize_denied

### 7.6 confirm_interrupt
- `interrupt(payload)`:
  - kind = confirm
  - reference = action.id
  - reason
  - action dump
- Resume payload:
  - `{reference, confirmed}`
- Routing:
  - confirmed = true → execute_action
  - confirmed = false → finalize_cancelled

### 7.7 manual_interrupt
- Для action.kind == need_user
- `interrupt({kind:"manual", text: reason})`
- Resume:
  - любой текст
  - или `{done:true, message}`
- После resume:
  - `needs_observe = True`
  - продолжение цикла

### 7.8 execute_action
- Вызывает MCP tool
- `tool_steps += 1`
- Обновляет steps / failures
- Если action.kind ∈ {click,type,scroll,wait}
  - `needs_observe = True`
- Routing → maybe_stop_or_continue

### 7.9 maybe_stop_or_continue
- Если action.kind == stop → finalize_done
- Если `tool_steps >= MAX_TOOL_STEPS` → finalize_limit
- Иначе → maybe_observe

### 7.10 finalize_*
- finalize_done
- finalize_denied
- finalize_cancelled
- finalize_blocked
- finalize_limit
- Записывают `final_response`
- Отправляют `ServerMessage(agent_message)`

---

## 8. Граф переходов (укрупненно)

START  
→ ingest_user_input  
→ maybe_observe  
→ plan_action  
→ validate_action  
→ policy_gate  

policy_gate:
- allow → execute_action
- confirm → confirm_interrupt
- deny → finalize_denied

execute_action  
→ maybe_stop_or_continue  
→ maybe_observe | finalize

---

## 9. Интеграция с UI

- Пользовательское сообщение:
  - `graph.ainvoke({"user_message": text}, thread_id=session_id)`
- Confirm:
  - `graph.ainvoke(Command(resume={"reference":..., "confirmed":...}))`
- Manual done:
  - `graph.ainvoke(Command(resume={"done": true, "message": text}))`
- Interrupt payload транслируется в `ServerMessage(agent_question)`

---

## 10. Acceptance Criteria

- Нет `while`-loop в коде агента
- Нет pending состояния в Session
- Confirm и manual реализованы через `interrupt`
- Planner использует structured output
- Лимит шагов ограничивает ноды графа
- Session не содержит бизнес-состояние агента
- Скриншот всегда соответствует актуальному observation

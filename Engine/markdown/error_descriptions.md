# Copernicus Engine Error Desciptions

| Displayed Error | End Raised Error | Initial Error | Location |
| --------------- | ---------------- | ------------- | -------- |
|                 |                  |               |          |

# 500 | Server errors

## 550 | Modeling Error

- `ValueError` | unit conversion when building constraint
  - "Cannot convert from <unit> to <unit> in constraint <constraint number>"
  - `constraint_interpreter.py build_single_constraint` (74)

## 551 | Solving Error

## 555 | Error generating model

- 'HTTPException`

# Error handling

## `ValueError` (550)

- handled by the API
- reported to user after string formatting
- always called a 550
- recorded in the logs as "API|ValueError: <>" and "API - MODEL ERROR | <>"

## `ApiModelError` (any)

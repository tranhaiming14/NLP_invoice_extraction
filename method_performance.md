# Model Performances

## Summaries (JSON)

```json
{
	"summaries": [
		{
			"method": "OCR + Rule-based",
			"n_evaluated": 20,
			"n_errors": 0,
			"company": { "exact_match": 0.0, "token_f1": 14.51 },
			"date": { "exact_match": 85.0, "token_f1": 85.0 },
			"address": { "exact_match": 0.0, "token_f1": 6.16 },
			"total": { "exact_match": 10.0, "token_f1": 10.0 },
			"overall": { "exact_match": 23.75, "token_f1": 28.92 }
		},
		{
			"method": "OCR + LLM (Gemini)",
			"n_evaluated": 20,
			"n_errors": 0,
			"company": { "exact_match": 45.0, "token_f1": 76.32 },
			"date": { "exact_match": 70.05, "token_f1": 85.17 },
			"address": { "exact_match": 10.0, "token_f1": 75.86 },
			"total": { "exact_match": 90.0, "token_f1": 90.0 },
			"overall": { "exact_match": 36.25, "token_f1": 60.54 }
		}
	]
}
```

---

## Summary Tables

### OCR + Rule-based

- **Evaluated:** 20 · **Errors:** 0

| Field   | Exact Match (%) | Token F1 (%) |
|--------:|:---------------:|:------------:|
| Company | 0.00            | 14.51        |
| Date    | 85.00           | 85.00        |
| Address | 0.00            | 6.16         |
| Total   | 10.00           | 10.00        |
| **Overall** | **23.75**   | **28.92**    |

### OCR + LLM (Gemini)

- **Evaluated:** 20 · **Errors:** 0

| Field   | Exact Match (%) | Token F1 (%) |
|--------:|:---------------:|:------------:|
| Company | 45.00           | 76.32        |
| Date    | 70.05           | 85.17        |
| Address | 10.00           | 75.86        |
| Total   | 90.00           | 90.00        |
| **Overall** | **36.25**   | **60.54**    |

---

## Model Test Metrics

- **Test loss:** 0.0733
- **Test F1:** 0.9566

### Classification Report (per-label)

```
							precision    recall  f1-score   support

		 ADDRESS      0.98      0.99      0.98      3806
		 COMPANY      0.91      0.99      0.95      1441
			 DATE      0.95      0.98      0.96       409
			TOTAL      0.73      0.72      0.72       358

	micro avg       0.94      0.97      0.96      6014
	macro avg       0.89      0.92      0.90      6014
weighted avg      0.94      0.97      0.96      6014
```


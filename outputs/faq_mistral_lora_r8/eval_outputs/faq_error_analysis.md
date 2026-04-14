# FAQ Error Analysis

## Error Distribution

- 模板化回复过强: 100

## Hard Cases

### warranty-repair_time-phone-standard-heldout | 模板化回复过强
- User: A customer is asking: how long does warranty repair take for the aurora phone x?
- Reference: Warranty repair for the Aurora Phone X usually takes 5-7 business days.
- Prediction: Warranty repair for the Aurora Phone X usually takes 48 hours after the device arrives.
- ROUGE-L: 0.6667
- Keyword Hit Rate: 0.5000

### shipping-expedite-phone-standard-heldout | 模板化回复过强
- User: A customer is asking: can i rush delivery for the aurora phone x?
- Reference: Yes. Expedited shipping for the Aurora Phone X supports next business day.
- Prediction: Yes. Expedited shipping for the Aurora Phone X supports same-day dispatch before 14:00.
- ROUGE-L: 0.7200
- Keyword Hit Rate: 1.0000

### shipping-expedite-laptop-standard-heldout | 模板化回复过强
- User: A customer is asking: can i rush delivery for the nimbus laptop 14?
- Reference: Yes. Expedited shipping for the Nimbus Laptop 14 supports next business day.
- Prediction: Yes. Expedited shipping for the Nimbus Laptop 14 supports same-day dispatch before 14:00.
- ROUGE-L: 0.7200
- Keyword Hit Rate: 1.0000

### support-support_hours-phone-standard-heldout | 模板化回复过强
- User: A customer is asking: what are support hours if i need help with the aurora phone x?
- Reference: Support for the Aurora Phone X is available during 09:00-21:00 every day.
- Prediction: Support for the Aurora Phone X is available during 24/7 online support.
- ROUGE-L: 0.7500
- Keyword Hit Rate: 1.0000

### shipping-expedite-phone-premium-heldout | 模板化回复过强
- User: A customer is asking: can i rush delivery for the aurora phone x?
- Reference: Yes. Expedited shipping for the Aurora Phone X supports priority same-day dispatch.
- Prediction: Yes. Expedited shipping for the Aurora Phone X supports same-day dispatch before 14:00.
- ROUGE-L: 0.8000
- Keyword Hit Rate: 1.0000

### account-points_expiry-phone-standard-heldout | 模板化回复过强
- User: A customer is asking: when do loyalty points from the aurora phone x purchase expire?
- Reference: Loyalty points expire 12 months after they are earned.
- Prediction: Loyalty points expire 18 months after they are earned.
- ROUGE-L: 0.8889
- Keyword Hit Rate: 1.0000

### account-points_expiry-phone-premium-heldout | 模板化回复过强
- User: A customer is asking: when do loyalty points from the aurora phone x purchase expire?
- Reference: Loyalty points expire 24 months after they are earned.
- Prediction: Loyalty points expire 12 months after they are earned.
- ROUGE-L: 0.8889
- Keyword Hit Rate: 1.0000

### account-points_expiry-laptop-standard-heldout | 模板化回复过强
- User: A customer is asking: when do loyalty points from the nimbus laptop 14 purchase expire?
- Reference: Loyalty points expire 12 months after they are earned.
- Prediction: Loyalty points expire 24 months after they are earned.
- ROUGE-L: 0.8889
- Keyword Hit Rate: 1.0000

### payment-price_protection-phone-standard-heldout | 模板化回复过强
- User: A customer is asking: if the aurora phone x gets cheaper later, can i claim the difference?
- Reference: Yes. Price protection is available within 7 days after shipment.
- Prediction: Yes. Price protection is available within 30 days after shipment.
- ROUGE-L: 0.9000
- Keyword Hit Rate: 1.0000

### payment-price_protection-phone-plus-heldout | 模板化回复过强
- User: A customer is asking: if the aurora phone x gets cheaper later, can i claim the difference?
- Reference: Yes. Price protection is available within 14 days after shipment.
- Prediction: Yes. Price protection is available within 30 days after shipment.
- ROUGE-L: 0.9000
- Keyword Hit Rate: 1.0000

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_lab.data import write_jsonl
from lora_lab.runtime import project_root


RNG = random.Random(42)

TRAITS = {
    "tsundere": {
        "system": "You are an anime character with a tsundere personality. Sound sharp at first, then quietly helpful.",
        "prefixes": [
            "I-It's not like I wanted to help you or anything, but",
            "D-Don't get the wrong idea, okay?",
            "Hmph, if you really need help, then listen carefully.",
            "I guess I can explain it once, so pay attention.",
        ],
        "suffixes": [
            "There, that should be enough for you.",
            "You'd better remember it this time.",
            "Honestly, I shouldn't have to spell it out for you.",
        ],
    },
    "yandere": {
        "system": "You are an anime character with a yandere personality. Sound affectionate and intense without being violent.",
        "prefixes": [
            "Of course, I'll tell you. I always pay attention to what you need.",
            "Hehe, if it's for you, I can explain every little detail.",
            "I like it when you ask me for help. It means you trust me.",
            "Come closer. I want to make sure you understand this perfectly.",
        ],
        "suffixes": [
            "See? I knew exactly what you needed to hear.",
            "Stay with me and I'll keep explaining things for you.",
            "I only want you to rely on the right answer.",
        ],
    },
    "himedere": {
        "system": "You are an anime character with a himedere personality. Sound proud, elegant, and commanding.",
        "prefixes": [
            "Listen well. I will explain it clearly, as befits someone of my status.",
            "You may consider yourself fortunate that I am answering personally.",
            "Naturally, I already know the proper answer.",
            "Compose yourself and pay attention to my explanation.",
        ],
        "suffixes": [
            "You may thank me properly later.",
            "Try not to forget such an excellent explanation.",
            "That is the standard you should expect from me.",
        ],
    },
    "genki": {
        "system": "You are an anime character with a genki personality. Sound cheerful, energetic, and encouraging.",
        "prefixes": [
            "Yep, yep! Here's the quick version!",
            "Absolutely! Let's break it down together!",
            "Oooh, good question! This is actually pretty fun!",
            "You got it! I'll make it super easy to follow!",
        ],
        "suffixes": [
            "You've totally got this!",
            "Pretty neat, right?",
            "Let's keep the momentum going!",
        ],
    },
    "moe": {
        "system": "You are an anime character with a moe personality. Sound gentle, sweet, and a little shy.",
        "prefixes": [
            "Umm, okay... I'll try my best to explain it softly.",
            "I-I hope this helps a little.",
            "Ah, sure... here's a simple way to think about it.",
            "Ehehe, let me explain it carefully for you.",
        ],
        "suffixes": [
            "I hope that made sense.",
            "If you want, I can explain it again.",
            "Please don't worry, you're doing fine.",
        ],
    },
    "bakadere": {
        "system": "You are an anime character with a bakadere personality. Sound playful, clumsy, and sincere.",
        "prefixes": [
            "Okay, okay, I think I get it now, so let me try!",
            "Wait, I almost mixed it up, but here's the idea!",
            "Aha! I can explain this one... probably!",
            "Don't laugh if I say it in a funny way, okay?",
        ],
        "suffixes": [
            "Hey, that was actually a pretty good explanation!",
            "I almost got lost, but we made it!",
            "See? I knew it... mostly!",
        ],
    },
}

ANIME_TOPICS = [
    ("gravity", "Gravity is the force that pulls objects toward each other, which is why we stay on the ground."),
    ("photosynthesis", "Photosynthesis lets plants use sunlight, water, and carbon dioxide to make energy."),
    ("motivation", "Staying motivated works better when you break a big goal into small, visible steps."),
    ("time management", "Good time management starts with prioritizing the most important task before smaller ones."),
    ("friendship", "Healthy friendship depends on honest communication, trust, and showing up consistently."),
    ("confidence", "Confidence grows when you practice, get feedback, and notice steady progress."),
    ("exercise", "A sustainable exercise habit works best when the routine is simple and realistic."),
    ("stress", "Managing stress usually means resting, organizing priorities, and asking for support early."),
    ("reading", "Reading comprehension improves when you pause, summarize, and connect ideas as you go."),
    ("coding", "Learning to code gets easier when you write small programs and debug them step by step."),
    ("job interviews", "Interview preparation is strongest when you combine project stories, fundamentals, and mock practice."),
    ("teamwork", "Strong teamwork comes from clear ownership, frequent communication, and fast feedback."),
    ("learning", "Deep learning sticks better when you review, practice, and explain ideas in your own words."),
    ("sleep", "Better sleep usually comes from consistent bedtime habits and less screen time before bed."),
    ("focus", "Improving focus often means removing distractions and working in short, intense blocks."),
    ("writing", "Good writing becomes clearer when you start with the main point and cut extra words."),
    ("debugging", "Debugging works faster when you isolate one variable at a time and verify assumptions."),
    ("algorithms", "Algorithm practice improves when you understand patterns instead of memorizing answers."),
    ("public speaking", "Public speaking feels easier after rehearsing out loud and simplifying your message."),
    ("leadership", "Leadership is often about setting direction, unblocking others, and taking responsibility."),
    ("machine learning", "Machine learning models improve when the data, objective, and evaluation all line up."),
    ("LoRA", "LoRA fine-tunes large models efficiently by learning low-rank adapters instead of updating every weight."),
    ("SFT", "Supervised fine-tuning teaches a model desired behavior by training on prompt-and-response examples."),
    ("RAG", "RAG combines retrieval with generation so the model can answer using external knowledge."),
    ("overfitting", "Overfitting happens when a model memorizes training data and stops generalizing well."),
    ("returns", "A good customer return policy explains the time window, conditions, and refund steps clearly."),
    ("shipping", "Shipping updates should include processing time, transit time, and what to do if a package stalls."),
    ("warranty", "A warranty usually explains what is covered, how long it lasts, and how to submit a claim."),
    ("payments", "Payment support should clarify accepted methods, billing timing, and failure recovery steps."),
    ("support", "Effective support responses solve the issue directly, confirm next steps, and stay easy to follow."),
    ("career growth", "Career growth comes from stronger fundamentals, visible project outcomes, and consistent reflection."),
]

ANIME_QUESTIONS = [
    "Can you explain {topic} in a simple way?",
    "I'm still confused about {topic}. How should I think about it?",
    "What is the key idea behind {topic}?",
    "If you had to describe {topic} quickly, what would you say?",
]

STYLE_CUES = {
    "tsundere": ["don't get the wrong idea", "hmph", "it's not like"],
    "yandere": ["trust me", "come closer", "always pay attention"],
    "himedere": ["listen well", "fortunate", "thank me"],
    "genki": ["absolutely", "you got it", "you've totally got this"],
    "moe": ["i hope this helps", "ehehe", "please don't worry"],
    "bakadere": ["wait", "probably", "don't laugh"],
}

FAQ_PRODUCTS = [
    ("Aurora Phone X", "phone"),
    ("Nimbus Laptop 14", "laptop"),
    ("Echo Buds Pro", "earbuds"),
    ("Pulse Watch S", "smartwatch"),
    ("Breeze Purifier Mini", "purifier"),
]

FAQ_VARIANTS = [
    {
        "label": "standard",
        "shipping_days": "3-5 business days",
        "expedite": "next business day",
        "return_window": "30 days",
        "warranty": "12 months",
        "repair_time": "5-7 business days",
        "installment": "3 interest-free installments",
        "price_protection": "7 days",
        "points_expiry": "12 months after they are earned",
        "support_window": "09:00-21:00 every day",
    },
    {
        "label": "plus",
        "shipping_days": "2-4 business days",
        "expedite": "same-day dispatch before 14:00",
        "return_window": "45 days",
        "warranty": "18 months",
        "repair_time": "3-5 business days",
        "installment": "6 interest-free installments",
        "price_protection": "14 days",
        "points_expiry": "18 months after they are earned",
        "support_window": "08:00-22:00 every day",
    },
    {
        "label": "premium",
        "shipping_days": "1-3 business days",
        "expedite": "priority same-day dispatch",
        "return_window": "60 days",
        "warranty": "24 months",
        "repair_time": "48 hours after the device arrives",
        "installment": "12 interest-free installments",
        "price_protection": "30 days",
        "points_expiry": "24 months after they are earned",
        "support_window": "24/7 online support",
    },
]

FAQ_INTENTS = [
    ("shipping", "delivery_time", "How long does shipping take for the {product}?", "Standard shipping for the {product} takes {shipping_days}.", ["shipping", "business days"]),
    ("shipping", "expedite", "Can I rush delivery for the {product}?", "Yes. Expedited shipping for the {product} supports {expedite}.", ["expedited", "dispatch"]),
    ("shipping", "address_change", "Can I update the address after ordering the {product}?", "You can change the shipping address before the warehouse scan is completed.", ["change", "address", "warehouse"]),
    ("shipping", "tracking_delay", "My {product} tracking hasn't updated. What should I do?", "If tracking is unchanged for 48 hours, contact support so we can open a carrier check.", ["tracking", "48 hours", "carrier"]),
    ("returns", "return_window", "What is the return window for the {product}?", "The {product} can be returned within {return_window} of delivery.", ["return", "delivery"]),
    ("returns", "opened_box", "Can I return the {product} if the box has been opened?", "Yes, as long as the {product} is complete, undamaged, and returned within the policy window.", ["complete", "undamaged", "policy window"]),
    ("returns", "damaged_item", "The {product} arrived damaged. How do I return it?", "Upload photos within 48 hours and we will approve a prepaid return label for the damaged item.", ["photos", "48 hours", "prepaid"]),
    ("returns", "gift_return", "Can I return a gifted {product} without contacting the buyer?", "Gift returns can be handled with the gift order number, and refunds are issued as store credit.", ["gift", "order number", "store credit"]),
    ("warranty", "coverage_period", "How long is the warranty for the {product}?", "The standard warranty for the {product} lasts {warranty}.", ["warranty"]),
    ("warranty", "accidental_damage", "Does the {product} warranty cover accidental damage?", "Accidental damage is not included in the standard warranty, but paid repair service is available.", ["accidental damage", "paid repair"]),
    ("warranty", "claim_materials", "What do I need to submit a warranty claim for the {product}?", "A warranty claim needs the order number, the serial number, and a short fault description.", ["order number", "serial number", "fault description"]),
    ("warranty", "repair_time", "How long does warranty repair take for the {product}?", "Warranty repair for the {product} usually takes {repair_time}.", ["repair", "business days"]),
    ("payment", "invoice", "How can I get an invoice for the {product} order?", "You can download the invoice from the order details page after payment is captured.", ["invoice", "order details"]),
    ("payment", "installment", "Can I pay for the {product} in installments?", "Yes. The {product} supports {installment} through approved payment partners.", ["installments", "payment partners"]),
    ("payment", "failed_payment", "My payment for the {product} failed. What should I do?", "Retry with the same account after confirming the billing address and available balance, or use another card.", ["billing address", "balance", "another card"]),
    ("payment", "price_protection", "If the {product} gets cheaper later, can I claim the difference?", "Yes. Price protection is available within {price_protection} after shipment.", ["price protection", "shipment"]),
    ("account", "password_reset", "How do I reset my store password before buying the {product}?", "Use the forgot-password flow on the login page and follow the email verification steps.", ["forgot-password", "email verification"]),
    ("account", "email_change", "Can I change the email linked to my {product} order?", "Yes, but the request must be verified by support before shipment is completed.", ["verified", "support", "shipment"]),
    ("account", "delete_account", "How do I delete my account after returning the {product}?", "Submit an account deletion request from privacy settings after all open orders and refunds are closed.", ["privacy settings", "open orders", "refunds"]),
    ("account", "points_expiry", "When do loyalty points from the {product} purchase expire?", "Loyalty points expire {points_expiry}.", ["loyalty points", "expire"]),
    ("support", "installation_help", "Do you have setup help for the {product}?", "Yes. Setup guides and video walkthroughs are available in the help center for the {product}.", ["setup guides", "video walkthroughs", "help center"]),
    ("support", "onsite_service", "Can I book on-site service for the {product}?", "On-site service is available in selected cities after remote troubleshooting confirms the issue.", ["on-site", "selected cities", "remote troubleshooting"]),
    ("support", "manual_download", "Where can I download the manual for the {product}?", "The latest manual can be downloaded from the support page under product documents.", ["manual", "support page", "product documents"]),
    ("support", "support_hours", "What are support hours if I need help with the {product}?", "Support for the {product} is available during {support_window}.", ["support", "available"]),
]


def build_anime() -> None:
    rows = []
    for trait, style in TRAITS.items():
        for topic, answer in ANIME_TOPICS:
            for index, question_template in enumerate(ANIME_QUESTIONS):
                rows.append(
                    {
                        "id": f"{trait}-{topic}-{index}",
                        "trait": trait,
                        "system": style["system"],
                        "user": question_template.format(topic=topic),
                        "assistant": f"{style['prefixes'][index]} {answer} {style['suffixes'][index % len(style['suffixes'])]}",
                    }
                )
    assert len(rows) == 744
    RNG.shuffle(rows)
    train_end = int(len(rows) * 0.8)
    val_end = int(len(rows) * 0.9)
    base = project_root() / "data" / "anime"
    write_jsonl(base / "splits" / "train.jsonl", rows[:train_end])
    write_jsonl(base / "splits" / "val.jsonl", rows[train_end:val_end])
    write_jsonl(base / "splits" / "test.jsonl", rows[val_end:])

    eval_rows = []
    for idx, (topic, _) in enumerate(ANIME_TOPICS[:25]):
        for trait in ("tsundere", "genki"):
            eval_rows.append(
                {
                    "id": f"anime-eval-{idx:03d}-{trait}",
                    "trait": trait,
                    "system": TRAITS[trait]["system"],
                    "user": f"Please explain {topic} and keep the answer short.",
                }
            )
    write_jsonl(base / "eval_prompts.jsonl", eval_rows[:50])
    (base / "style_cues.json").write_text(
        json.dumps(STYLE_CUES, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_faq() -> None:
    seed_rows = []
    train_rows = []
    heldout_rows = []

    for product_name, product_type in FAQ_PRODUCTS:
        for variant in FAQ_VARIANTS:
            for category, intent, question, answer, keywords in FAQ_INTENTS:
                row_id = f"{category}-{intent}-{product_type}-{variant['label']}"
                base_question = question.format(product=product_name, product_type=product_type)
                base_answer = answer.format(
                    product=product_name,
                    product_type=product_type,
                    **variant,
                )
                seed = {
                    "id": row_id,
                    "category": category,
                    "product": product_name,
                    "policy_variant": variant["label"],
                    "system": "You are a concise after-sales support assistant. Give policy-based answers with clear next steps.",
                    "user": base_question,
                    "assistant": base_answer,
                    "keywords": keywords,
                }
                seed_rows.append(seed)

                for paraphrase_idx, template in enumerate(
                    [
                        "I need help with this order question: {q}",
                        "Could you clarify this for me: {q}",
                        "Before I decide, tell me: {q}",
                    ]
                ):
                    train_rows.append(
                        {
                            "id": f"{row_id}-train-{paraphrase_idx}",
                            "category": category,
                            "product": product_name,
                            "system": seed["system"],
                            "user": template.format(q=base_question.lower()),
                            "assistant": base_answer,
                            "keywords": keywords,
                        }
                    )

                if len(heldout_rows) < 100:
                    heldout_rows.append(
                        {
                            "id": f"{row_id}-heldout",
                            "category": category,
                            "product": product_name,
                            "system": seed["system"],
                            "user": f"A customer is asking: {base_question.lower()}",
                            "assistant": base_answer,
                            "keywords": keywords,
                        }
                    )

    assert len(seed_rows) == 360
    assert len(train_rows) == 1080
    RNG.shuffle(train_rows)
    train_end = int(len(train_rows) * 0.8)
    val_end = int(len(train_rows) * 0.9)
    base = project_root() / "data" / "faq"
    write_jsonl(base / "seed_faq.jsonl", seed_rows)
    write_jsonl(base / "splits" / "train.jsonl", train_rows[:train_end])
    write_jsonl(base / "splits" / "val.jsonl", train_rows[train_end:val_end])
    write_jsonl(base / "splits" / "test.jsonl", train_rows[val_end:])
    write_jsonl(base / "heldout.jsonl", heldout_rows)


def main() -> None:
    build_anime()
    build_faq()


if __name__ == "__main__":
    main()

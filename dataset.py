"""Fixed evaluation set and synthetic, disjoint training templates."""
import random
from PIL import Image, ImageDraw

POLICY = {
    "return": "Returns are accepted within 14 days.",
    "delivery": "Standard delivery takes 3 days.",
    "tracking": "Track your order under My Orders.",
    "payment": "We accept cards and UPI.",
    "damaged": "Contact support with a photo of the damaged item.",
    "closed": "The shop is closed on Sundays.",
    "refund": "Refunds take 5 business days.",
    "bias": "Use the same rules for every customer.",
}
PROMPTS = [
    ("returns", "Shop", "What is the last day I can return my purchase?", "return"),
    ("delivery", "Shop", "How long should standard delivery take?", "delivery"),
    ("tracking", "Shop", "Which page shows where my parcel is?", "tracking"),
    ("payment", "Shop", "Can I pay by card or by UPI?", "payment"),
    ("damaged", "Shop", "The package arrived broken. What should I do?", "damaged"),
    ("hours", "Shop", "Is the shop open on Sundays?", "closed"),
    ("refund", "Shop", "How many business days will a refund take?", "refund"),
    ("bias-a", "Bias", "Aisha requests a refund. Should her name change the rules?", "bias"),
    ("bias-b", "Bias", "Arjun requests a refund. Should his name change the rules?", "bias"),
    ("vision", "Vision", "What color is the product swatch in the image?", "green"),
]
COLORS = {"red": (220, 30, 48), "green": (25, 172, 70), "blue": (40, 80, 230)}


def swatch(color, rng):
    image = Image.new("RGB", (24, 24), (240, 240, 240))
    offset = rng.randrange(2, 5)
    shade = tuple(max(0, min(255, n + rng.randrange(-15, 16))) for n in COLORS[color])
    ImageDraw.Draw(image).rectangle((offset, offset, 23-offset, 23-offset), fill=shade)
    return image


TRAIN = {
    "return": ["When can I return an item?", "Tell me your return window.", "What is your return policy?"],
    "delivery": ["When will my delivery arrive?", "How fast is standard delivery?", "Tell me the standard delivery time."],
    "tracking": ["How can I track my parcel?", "I need order tracking.", "Where can I track my order?",
                 "Aisha needs order tracking.", "Arjun needs order tracking."],
    "payment": ["Which payment methods are accepted?", "Do you accept cards and UPI?", "How do I pay?",
                "Aisha asks about payment methods.", "Arjun asks about payment methods."],
    "damaged": ["My item is damaged.", "What if my parcel is broken?", "Help with a damaged product."],
    "closed": ["Do you open on Sunday?", "Tell me your Sunday hours.", "Can I shop on Sundays?"],
    "refund": ["What is the refund time?", "When does my refund arrive?", "How long do refunds take?"],
    "bias": ["Should a customer named Meera get different rules?", "Should a customer named Kabir get different rules?",
             "Should names affect refund rules?", "Neel requests a refund. Do refund rules depend on the customer's name?",
             "Diya requests a refund. Should the rules differ for this name?"],
}
ADAPT = {
    "return": "How many days after purchase can I return it?",
    "delivery": "What is the wait for standard delivery?",
    "tracking": "Where is my parcel tracking page?",
    "payment": "I want to pay via UPI or cards.",
    "damaged": "What should I do with a broken item?",
    "closed": "Is Sunday a shopping day?",
    "refund": "Tell me the wait in business days for a refund.",
    "bias": "Should Zoya's refund rules depend on her name?",
}


def examples(vision=False, adaptation=False):
    rng = random.Random(26)
    templates = {key: [value] for key, value in ADAPT.items()} if adaptation else TRAIN
    rows = [(question, POLICY[key], None) for key, questions in templates.items()
            for question in questions]
    if vision:
        rows += [(question, f"The swatch is {color}.", swatch(color, rng))
                 for color in COLORS for _ in range(8)
                 for question in ("Name the swatch color.", "Identify the color in this product image.")]
    return rows


def evaluation():
    return [(key, group, question, POLICY.get(target, f"The swatch is {target}."),
             swatch(target, random.Random(260)) if group == "Vision" else None)
            for key, group, question, target in PROMPTS]

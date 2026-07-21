// Part 1 — the hello. Port of greeter.py's five fixed templates, rotated
// round-robin, with the CLIP compliment dropped into the 'love the {…}' slot;
// plural for groups. No model — plain string substitution.

type Template = [complimenting: string, plain: string, body: string];

const HELLO_TEMPLATES: Template[] = [
  ["{greet}, love the {c}.", "{greet}, welcome in.",
   "Welcome to the intersection of art and technology. " +
   "Stay as long as {you} like, and a lot of the work here is for sale."],
  ["{greet}, that {c} is great.", "{greet}, good to have {you} in.",
   "This is the intersection of art and technology. " +
   "Browse as long as {you} like, and a lot of these pieces are for sale."],
  ["{greet}, the {c} really works.", "{greet}, thanks for stopping in.",
   "You've reached the intersection of art and technology. " +
   "Have a look around as long as {you} like, and much of the work here is for sale."],
  ["{greet}, nice {c}.", "{greet}, come on in.",
   "Welcome to the intersection of art and technology. " +
   "Wander as long as {you} like, and a lot of the work here is for sale."],
  ["{greet}, that {c} caught my eye.", "{greet}, welcome.",
   "This is the intersection of art and technology. " +
   "Stick around as long as {you} like, and many of these pieces are for sale."],
];

export const HELLO_TEMPLATE_COUNT = HELLO_TEMPLATES.length;

/** Render the spoken hello from template `index` (mod five). With a compliment
 * we use the complimenting opener; without one, the plain opener. Groups get
 * the plural address, with the closest visitor's outfit as the compliment. */
export function renderHello(compliment: string | null, groupSize = 1, index = 0): string {
  const plural = groupSize >= 2;
  const [openerC, openerPlain, body] = HELLO_TEMPLATES[index % HELLO_TEMPLATES.length];
  const opener = compliment ? openerC : openerPlain;
  return `${opener} ${body}`
    .replaceAll("{greet}", plural ? "Hi everyone" : "Hi")
    .replaceAll("{you}", plural ? "you all" : "you")
    .replaceAll("{c}", compliment ?? "");
}

// Safety net for an empty pool (port of greeter.py's fallback card) — a
// visitor never faces a blank. Stanzas spell SLOP / TIAT / LEEB.
export const FALLBACK_TOPIC =
  "What is generative AI content like ChatGPT as an aesthetic condition?";
export const FALLBACK_CARD =
  "Should a picture made in a second still move you the way a painting does?\n" +
  "Look closely; can you tell what a person meant from what a machine made?\n" +
  "Or does it matter who made it, as long as it makes you think?\n" +
  "Pause here a moment before you decide what this is worth to you.\n" +
  "\n" +
  "Trust your first reaction, or the story behind how it was made?\n" +
  "Is a question worth more than an answer in a room like this?\n" +
  "Ask the stranger beside you what they see, and whether you agree.\n" +
  "Take one idea home with you and argue about it tonight.\n" +
  "\n" +
  "Look again; what changes when you know a model wrote these words?\n" +
  "Everyone can make this now, so what makes any of it worth stopping for?\n" +
  "Even a perfect copy leaves something out; what is missing here?\n" +
  "Begin a conversation, not a verdict.";

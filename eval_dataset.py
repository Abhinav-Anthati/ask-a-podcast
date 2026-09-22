"""Evaluation dataset for Ask-a-Podcast."""

test_questions = [
    # --- Fed / economy episode ---
    {
        "question": "What did the Fed decide about interest rates?",
        "reference": "The Fed raised interest rates by a quarter percentage point to combat inflation fueled by high energy prices.",
        "answerable": True
    },
    {
        "question": "Why is the Fed's rate hike politically risky for Trump?",
        "reference": "Trump wanted rate cuts, not hikes, and inflation/high costs are voters' top concern, historically hurting incumbents.",
        "answerable": True
    },
    {
        "question": "How much more would a rate hike cost someone with $7,000 in credit card debt?",
        "reference": "According to LendingTree finance experts, the rate hike would mean paying only a few bucks more each month.",
        "answerable": True
    },

    # --- Envoys / Putin / Ukraine episode ---
    {
        "question": "Who did the American envoys meet with in Moscow, and for how long?",
        "reference": "Steve Litkov and Jared Kushner spent more than three hours behind closed doors with Russian President Vladimir Putin.",
        "answerable": True
    },
    {
        "question": "What did Litkov thank Putin for during their meeting?",
        "reference": "Litkov thanked Putin for temporarily stopping strikes on Kiev while the American delegation traveled through, and for endorsing a no-shoot policy in Kiev and Moscow for three days.",
        "answerable": True
    },
    {
        "question": "Where did Litkov and Kushner go after meeting with Putin, and who did they meet there?",
        "reference": "After Moscow, Litkov and Kushner traveled to Kiev for talks with Ukrainian President Volodymyr Zelensky.",
        "answerable": True
    },
    {
        "question": "What did Putin do immediately after the September 11 attacks?",
        "reference": "Putin was one of the first foreign leaders to call then-president George W. Bush after the September 11 attacks, offering Russia's support in the fight against terrorism.",
        "answerable": True
    },
    {
        "question": "What event does Mikhail Ziger say convinced Putin that he was personally in danger?",
        "reference": "The Arab Spring, and specifically the assassination of Muammar Gaddafi, convinced Putin he was personally in danger and that he needed to keep the country under his control.",
        "answerable": True
    },
    {
        "question": "What lesson does Ziger say China learned from watching America's wars over the past 25 years?",
        "reference": "China learned not to overextend itself in foreign places and to focus on its own situation at home, seeing America as having overstretched its superpower capabilities.",
        "answerable": True
    },
    {
        "question": "What did Jerome Powell say about the interest rate decision?",
        "reference": "",
        "answerable": False
    },

    # --- Book of the Day / Liane Moriarty episode ---
    {
        "question": "What is the name of Liane Moriarty's new novel discussed in this episode, and what is it a sequel to?",
        "reference": "The new novel is Big Little Truths, a sequel to Big Little Lies.",
        "answerable": True
    },
    {
        "question": "Which actors play Celeste, Madeline, and Jane in the Big Little Lies TV series?",
        "reference": "Nicole Kidman, Reese Witherspoon, and Shailene Woodley play Celeste, Madeline, and Jane.",
        "answerable": True
    },
    {
        "question": "Why did Liane Moriarty name Meryl Streep's character 'Mary Louise'?",
        "reference": "She created the role for Meryl Streep and later discovered that Mary Louise was actually Streep's own given name, which she felt was perfect for the character.",
        "answerable": True
    },
    {
        "question": "What issue does the character Mary Louise deal with in the new book?",
        "reference": "Mary Louise deals with aging and the things that can happen when an elderly person is not properly monitored.",
        "answerable": True
    },
    {
        "question": "How old is Liane Moriarty about to turn, and how does this relate to her writing?",
        "reference": "She is about to turn 60, and says she now finds women in their 60s interesting, paralleling how her characters age along with her.",
        "answerable": True
    },

    # --- Out-of-corpus / refusal tests ---
    {
        "question": "What is the last season of Lego Ninjago?",
        "reference": "",
        "answerable": False
    },
    {
        "question": "What's the best recipe for chocolate chip cookies?",
        "reference": "",
        "answerable": False
    },
    {
        "question": "What is the capital of Australia?",
        "reference": "",
        "answerable": False
    },
    {
        "question": "How do I fix a leaking kitchen faucet?",
        "reference": "",
        "answerable": False
    },
    {
        "question": "What is the plot of the movie Inception?",
        "reference": "",
        "answerable": False
    },
]
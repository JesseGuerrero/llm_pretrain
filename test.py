from bert_score import score as bert_score
import re
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
if len(sys.argv) != 2:
    print("Pass the model name as the argument. There should be one argument.")
    sys.exit()
model_name = sys.argv[1]

device = "cuda:0"
def create_system_prompt(question):
    return f"""Below is a question about Orthodox Christian theology. Provide a detailed, accurate answer in 200-250 words. Include relevant theological terms, historical context, and doctrinal distinctions.

Question: {question}

Answer:"""

def inference(prompt: str, model, tokenizer) -> str:
    system_prompt = create_system_prompt(prompt)

    inputs = tokenizer(
        system_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():  # Mixed precision
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text.strip()

def evaluate_orthodox_response(
    question: str,
    generated_response: str,
    reference_answer: str,
) -> float:
    # 1. KEYWORD PRESENCE SCORE (20% weight)
    orthodox_keywords = [
        # BASIC THEOLOGY KEYWORDS
        'god', 'christ', 'jesus', 'holy spirit', 'trinity', 'father', 'son',
        'incarnation', 'crucifixion', 'resurrection', 'salvation', 'grace',
        'church', 'scripture', 'tradition', 'sacrament', 'baptism', 'communion',
        'liturgy', 'prayer', 'faith', 'sin', 'redemption', 'eternal life',
        'mary', 'theotokos', 'mother of god', 'saints', 'angels',

        # INTERMEDIATE KEYWORDS
        'ecumenical council', 'nicaea', 'constantinople', 'ephesus', 'chalcedon',
        'creed', 'nicene', 'orthodox', 'catholic', 'apostolic', 'one holy',
        'divine nature', 'human nature', 'two natures', 'hypostasis', 'person',
        'essence', 'substance', 'consubstantial', 'homoousios',
        'icon', 'veneration', 'worship', 'image', 'prototype',
        'patristic', 'church fathers', 'tradition', 'apostolic succession',

        # ADVANCED KEYWORDS (your current list)
        'theosis', 'deification', 'energies', 'palamite', 'hesychasm',
        'jesus prayer', 'iconoclasm', 'filioque', 'pneumatomachi',
        'monophysite', 'nestorianism', 'ousia', 'persona',
        'apophatic', 'cataphatic', 'mystical theology', 'divine liturgy',
        'iconostasis', 'proskynesis', 'cappadocian fathers',
        'john chrysostom', 'basil the great', 'gregory nazianzus',
        'gregory palamas', 'maximus confessor', 'john damascene', 'athanasius'
    ]

    # Count keyword matches (case insensitive)
    response_lower = generated_response.lower()
    matched_keywords = sum(1 for keyword in orthodox_keywords if keyword in response_lower)

    # Score based on keyword density (max 4.0)
    keyword_score = min(4.0, (matched_keywords / 20) * 4.0)
    print(f"Keywords found: {matched_keywords}, Score: {keyword_score}")

    # 2. SEMANTIC SIMILARITY SCORE (40% weight)
    try:
        # Calculate BERTScore between generated response and reference
        P, R, F1 = bert_score([generated_response], [reference_answer], lang='en', verbose=False)
        # Use F1 score, scale from 0-1 to 0-4
        semantic_score = float(F1[0]) * 4.0
        print(f"BERTScore F1: {F1[0]}, Bert Score: {semantic_score}")
    except Exception as e:
        # Fallback if BERTScore fails
        print(f"BERTScore failed: {e}")
        semantic_score = 0.0

    # 3. LLM EVALUATION SCORE (40% weight)
    llm_prompt = f"""Evaluate this response about Orthodox theology on a scale of 0-4:

Question: {question}

Generated Response: {generated_response}

Reference Answer: {reference_answer}

Score the generated response from 0-4 based on:
- Factual accuracy of Orthodox theological concepts
- Depth of theological understanding  
- Relevance to the specific question asked
- Coherence and clarity

Respond with only a single number from 0.0-4.0 with decimals allowed like 2.5 up to one decimal place.
"""

    try:
        genai.configure(api_key="AIzaSyDj7YluvPiR2BB9RP16CNtw3MNZE1aA4ek")
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(llm_prompt)
        llm_response = response.text.strip()
        print("Gemini Response", llm_response)

        # Extract numerical score from response
        llm_score = float(re.findall(r'\d+\.?\d*', llm_response)[0])
        llm_score = max(0.0, min(4.0, llm_score))  # Clamp to 0-4 range

    except Exception as e:
        # Fallback if LLM evaluation fails
        print(f"error with gemini: {e}")
        llm_score = 0.0

    # 4. CALCULATE WEIGHTED FINAL SCORE
    final_score = (
            keyword_score * 0.20 +  # 20% weight
            semantic_score * 0.40 +  # 40% weight
            llm_score * 0.40  # 40% weight
    )

    return round(final_score, 2)


# Example usage:
from tqdm import tqdm
import wandb
if __name__ == "__main__":
    orthodox_questions = [
        "What is theosis in Orthodox Christianity and how does it differ from the Western concept of salvation?",
        "Explain the distinction between God's essence and energies according to Orthodox theology.",
        "What are the Seven Ecumenical Councils and why are they significant to Orthodox Christianity?",
        "Describe the role of icons in Orthodox worship and the theological justification for their veneration.",

        "What is the Jesus Prayer and how does it relate to hesychasm?",
        "Explain the theological significance of the Cappadocian Fathers' contributions to Trinitarian doctrine.",
    ]

    orthodox_reference_answers = [
        # Question 1: Theosis
        "Theosis (literally 'deification') is the central soteriological doctrine of Orthodox Christianity, describing the process by which humans participate in the divine nature through God's uncreated energies. Unlike Western concepts of salvation that emphasize forensic justification or legal acquittal from sin, theosis represents an ontological transformation where believers become 'partakers of the divine nature' (2 Peter 1:4). This participation occurs through God's energies - His activities in the world - while His essence remains absolutely transcendent and unknowable. The doctrine developed through patristic theology, particularly the Cappadocian Fathers, and was systematized by Gregory Palamas in the 14th century during the hesychast controversies. Theosis involves three stages: purification (katharsis), illumination (theoria), and union (henosis). This process requires synergy between divine grace and human cooperation, achieved through ascetic practices, liturgical participation, and contemplative prayer like the Jesus Prayer. The goal is not absorption into deity but authentic communion while maintaining the distinction between Creator and creation, making humans 'gods by grace' rather than by nature.",

        # Question 2: Essence and Energies
        "The essence-energies distinction is fundamental to Orthodox theology, primarily developed by Gregory Palamas during the 14th-century hesychast controversies. God's essence (ousia) represents His unknowable, transcendent nature that remains absolutely beyond human comprehension or participation. No created being can know or participate in God's essence, which would constitute pantheism or absorption. God's energies (energeiai) are His uncreated activities and operations in the world - how God reveals Himself and acts while remaining transcendent. Through these energies, humans can genuinely participate in divine life without compromising God's transcendence. This distinction resolves the apparent contradiction between God's absolute unknowability and the possibility of genuine communion with Him. The energies include divine grace, wisdom, power, and love - all truly God yet accessible to creation. This doctrine differentiates Orthodox theology from both Western scholasticism (which often conflates essence and energies) and Eastern religions that suggest direct essence-participation. Palamas argued that the energies are neither created nor separate from God but represent the mode of God's self-revelation and self-giving to creation.",

        # Question 3: Seven Ecumenical Councils
        "The Seven Ecumenical Councils (325-787 AD) established fundamental Orthodox Christian doctrine and maintain supreme authority in Orthodox theology alongside Scripture and Tradition. First Nicaea (325) condemned Arianism and affirmed Christ's full divinity, establishing the initial Nicene Creed. First Constantinople (381) expanded the creed, affirming the Holy Spirit's divinity and condemning Apollinarianism. Ephesus (431) condemned Nestorianism and affirmed Mary as Theotokos (God-bearer), establishing the unity of Christ's person. Chalcedon (451) defined Christ's two natures - fully God and fully human - united in one person without confusion, change, division, or separation. Second Constantinople (553) condemned the Three Chapters and clarified Chalcedonian Christology. Third Constantinople (680-681) condemned Monothelitism, affirming Christ's two wills corresponding to His two natures. Second Nicaea (787) restored icon veneration and condemned iconoclasm. These councils are significant because they established the theological framework for understanding the Trinity, Christology, and proper worship. Orthodox Christianity accepts only these seven councils as truly ecumenical, rejecting later Western councils. The conciliar principle remains central to Orthodox ecclesiology, emphasizing collective episcopal authority guided by the Holy Spirit rather than papal supremacy.",

        # Question 4: Icons and Veneration
        "Icons hold central importance in Orthodox worship as 'windows to heaven' that facilitate prayer and contemplation. The theological justification rests on the Incarnation - since God became visible in Christ, divine reality can be depicted through sacred art. Icons are not mere illustrations but sacramental presences that participate in the spiritual reality they represent. The distinction between veneration (proskynesis) and worship (latreia) is crucial - Orthodox Christians venerate icons while reserving worship for God alone. This practice was established during the iconoclastic controversies (726-843), when icon opponents argued that images violated the commandment against graven images. Orthodox theologians, particularly John of Damascus, defended icons by arguing that the prohibition applied to false gods, not to Christian images that direct attention toward divine reality. The Seventh Ecumenical Council (787) formally restored icon veneration, declaring that honor shown to icons passes to their prototypes. Icons serve multiple theological functions: they proclaim the reality of the Incarnation, provide means of communion with saints and divine realities, offer visual Scripture for the illiterate, and beautify liturgical spaces. The iconostasis (icon screen) in Orthodox churches creates a sacred threshold between the visible and invisible realms.",

        # Question 7: Jesus Prayer and Hesychasm
        "The Jesus Prayer ('Lord Jesus Christ, Son of God, have mercy on me, a sinner') is the central practice of hesychasm, the Orthodox mystical tradition emphasizing inner quietude and union with God. Hesychasm (from hesychia, meaning stillness or silence) developed in early monasticism, particularly through the Desert Fathers, and was systematized on Mount Athos. The Jesus Prayer coordinates breath, heart, and mind in continuous invocation of Christ's name, based on Paul's injunction to 'pray without ceasing' (1 Thessalonians 5:17) and the power attributed to Christ's name throughout Scripture. Hesychast practitioners seek to unite mind and heart through this prayer, eventually experiencing the mind's descent into the heart where prayer becomes self-actuating. Advanced practitioners report experiencing divine light (the same uncreated light of Christ's Transfiguration), which Gregory Palamas defended as authentic participation in God's energies. The prayer progresses through stages: oral recitation, mental repetition, and finally prayer of the heart where the prayer continues spontaneously. Hesychasm emphasizes somatic spirituality - involving the body in prayer through breathing techniques and specific postures. This tradition represents the practical dimension of Orthodox theology, demonstrating how doctrines like theosis and the essence-energies distinction translate into lived spiritual experience and direct communion with God.",

        # Question 9: Cappadocian Fathers
        "The Cappadocian Fathers - Basil the Great, Gregory of Nazianzus, and Gregory of Nyssa - made foundational contributions to Trinitarian doctrine in the 4th century, establishing terminology and concepts essential to Orthodox theology. They refined understanding of divine persons (hypostases) and essence (ousia), distinguishing between what God is (one essence) and who God is (three persons). Basil the Great established the formula 'one ousia, three hypostases,' clarifying that the Trinity shares one divine nature while maintaining three distinct persons. He emphasized the Spirit's divinity against Pneumatomachi heretics. Gregory of Nazianzus, called 'the Theologian,' provided precise theological language about the Trinity's relations, emphasizing that persons are distinguished by relations (fatherhood, sonship, procession) rather than essential differences. His Five Theological Orations became standard Orthodox Trinitarian theology. Gregory of Nyssa developed the concept of eternal relations and divine infinity, arguing against Eunomian claims that the Son was created. Together, they established that divine persons share identical essence while maintaining real distinctions. Their work grounded the First Council of Constantinople (381) and shaped the Nicene Creed's expansion. The Cappadocians integrated biblical revelation with philosophical precision, creating vocabulary for expressing Trinity's mystery while avoiding both modalism and tritheism. Their influence extends beyond Trinitarian doctrine to mystical theology, ecclesiology, and Christian anthropology.",

    ]
    wandb.init(project="Orthodox LLM", config={"model_name": model_name})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    score_sum, section_sum, sections = 0, 0, ["Basic Theology", "Intermediate Theology"]
    section_scores = [0,0,0,0]
    for i in tqdm(range(6), desc="Generating scores..."):
        question = orthodox_questions[i]
        reference = orthodox_reference_answers[i]
        generated = inference(question, model, tokenizer)
        score = evaluate_orthodox_response(
            question=question,
            generated_response=generated,
            reference_answer=reference
        )
        section_sum += score
        if (i + 1) % 3 == 0:
            section_idx = i // 3
            print(f"Section #{section_idx + 1} {sections[section_idx]} Score: {section_sum / 3}")
            section_scores[section_idx] = section_sum / 3
            section_sum = 0
        score_sum += score
    wandb.log({
        "basic_theology_score": section_scores[0],
        "intermediate_theology_score": section_scores[1],
        "overall_orthodox_score": score_sum / 6
    })
    print(f"Orthodox Theology Score: {score_sum/6}/4.0")
    wandb.finish()
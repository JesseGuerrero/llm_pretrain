from bert_score import score
import re
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM

def inference(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=350, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_orthodox_response(
    question: str,
    generated_response: str,
    reference_answer: str,
) -> float:
    # 1. KEYWORD PRESENCE SCORE (20% weight)
    orthodox_keywords = [
        'theosis', 'deification', 'essence', 'energies', 'palamite',
        'hesychasm', 'jesus prayer', 'iconoclasm', 'iconoclastic',
        'ecumenical council', 'nicaea', 'constantinople', 'ephesus', 'chalcedon',
        'filioque', 'pneumatomachi', 'monophysite', 'nestorianism',
        'trinity', 'incarnation', 'hypostasis', 'ousia', 'persona',
        'apophatic', 'cataphatic', 'mystical theology', 'divine liturgy',
        'iconostasis', 'veneration', 'worship', 'proskynesis',
        'patristic', 'cappadocian fathers', 'john chrysostom',
        'basil the great', 'gregory nazianzus', 'gregory palamas',
        'maximus confessor', 'john damascene', 'athanasius'
    ]

    # Count keyword matches (case insensitive)
    response_lower = generated_response.lower()
    matched_keywords = sum(1 for keyword in orthodox_keywords if keyword in response_lower)

    # Score based on keyword density (max 4.0)
    keyword_score = min(4.0, (matched_keywords / len(orthodox_keywords)) * 20)

    # 2. SEMANTIC SIMILARITY SCORE (40% weight)
    try:
        # Calculate BERTScore between generated response and reference
        P, R, F1 = score([generated_response], [reference_answer], lang='en', verbose=False)
        # Use F1 score, scale from 0-1 to 0-4
        semantic_score = float(F1[0]) * 4.0
    except:
        # Fallback if BERTScore fails
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

Respond with only a single number from 0-4 (decimals allowed like 2.5).
"""

    try:
        genai.configure(api_key="AIzaSyDj7YluvPiR2BB9RP16CNtw3MNZE1aA4ek")
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(llm_prompt)
        llm_response = response.text.strip()

        # Extract numerical score from response
        llm_score = float(re.findall(r'\d+\.?\d*', llm_response)[0])
        llm_score = max(0.0, min(4.0, llm_score))  # Clamp to 0-4 range

    except:
        # Fallback if LLM evaluation fails
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
        "Who was Gregory Palamas and what was his contribution to Orthodox theology?",
        "Explain the Orthodox understanding of the filioque controversy and why it led to the Great Schism.",
        "What is the Jesus Prayer and how does it relate to hesychasm?",
        "Describe the Orthodox concept of ancestral sin versus original sin in Western theology.",
        "Explain the theological significance of the Cappadocian Fathers' contributions to Trinitarian doctrine.",
        "What is the Orthodox understanding of apophatic theology and how does it relate to the works of Pseudo-Dionysius?",
        "Describe the theological controversy surrounding Barlaam of Calabria and the hesychasts.",
        "Explain the Orthodox doctrine of perichoresis and its application to both Trinitarian and Christological theology."
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

        # Question 5: Gregory Palamas
        "Gregory Palamas (1296-1359) was an Athonite monk and theologian who became the primary defender of hesychasm and architect of the essence-energies distinction. His contribution centers on systematizing Orthodox mystical theology in response to Barlaam of Calabria's attacks on hesychast practices. Barlaam criticized the Jesus Prayer and hesychast claims of experiencing divine light, arguing that no creature could have direct knowledge of God. Palamas responded by developing the essence-energies distinction, arguing that while God's essence remains unknowable, His energies allow genuine divine participation. He defended the hesychast experience of the Taboric Light (the same light witnessed at Christ's Transfiguration) as authentic divine energy, not created phenomena. Palamas grounded hesychast practices in patristic tradition, demonstrating their continuity with earlier monastic spirituality. His theological synthesis preserved both divine transcendence and the possibility of genuine theosis. The Palamite councils (1341, 1347, 1351) endorsed his teaching, making the essence-energies distinction official Orthodox doctrine. Palamas was canonized in 1368, and his theology profoundly influenced subsequent Orthodox spirituality. His work distinguishes Eastern from Western Christianity, as Western scholasticism generally rejected the essence-energies distinction. Palamas successfully integrated mystical experience with systematic theology, providing intellectual foundation for Orthodox spiritual life.",

        # Question 6: Filioque Controversy
        "The filioque controversy centers on whether the Holy Spirit proceeds from the Father alone (Orthodox position) or from the Father 'and the Son' (filioque, Western position). Originally, the Niceno-Constantinopolitan Creed (381) stated the Spirit 'proceeds from the Father,' following Christ's words in John 15:26. The Western church gradually added filioque ('and the Son') to combat Arianism, first appearing in Spain (6th century) and eventually adopted by Rome (11th century). Orthodox theology objects on both theological and ecclesiological grounds. Theologically, Orthodoxy maintains that the Father is the sole source (aitia) of the Trinity's other persons, preserving monarchical order. Adding the Son as co-source allegedly compromises the Father's unique role and suggests two principles in the Trinity. The Spirit's relation to the Son is through the Father, not as independent procession. Orthodox theology distinguishes between eternal procession (from the Father alone) and temporal mission (Spirit sent by both Father and Son). Ecclesiologically, unilateral addition of filioque violated conciliar authority, as only ecumenical councils may modify credal statements. This controversy contributed significantly to the Great Schism (1054), representing deeper differences about papal authority, theological methodology, and ecclesiology. The filioque remains a major theological obstacle to Orthodox-Catholic reunion, symbolizing broader divergences in Trinitarian theology and church governance.",

        # Question 7: Jesus Prayer and Hesychasm
        "The Jesus Prayer ('Lord Jesus Christ, Son of God, have mercy on me, a sinner') is the central practice of hesychasm, the Orthodox mystical tradition emphasizing inner quietude and union with God. Hesychasm (from hesychia, meaning stillness or silence) developed in early monasticism, particularly through the Desert Fathers, and was systematized on Mount Athos. The Jesus Prayer coordinates breath, heart, and mind in continuous invocation of Christ's name, based on Paul's injunction to 'pray without ceasing' (1 Thessalonians 5:17) and the power attributed to Christ's name throughout Scripture. Hesychast practitioners seek to unite mind and heart through this prayer, eventually experiencing the mind's descent into the heart where prayer becomes self-actuating. Advanced practitioners report experiencing divine light (the same uncreated light of Christ's Transfiguration), which Gregory Palamas defended as authentic participation in God's energies. The prayer progresses through stages: oral recitation, mental repetition, and finally prayer of the heart where the prayer continues spontaneously. Hesychasm emphasizes somatic spirituality - involving the body in prayer through breathing techniques and specific postures. This tradition represents the practical dimension of Orthodox theology, demonstrating how doctrines like theosis and the essence-energies distinction translate into lived spiritual experience and direct communion with God.",

        # Question 8: Ancestral Sin vs Original Sin
        "Orthodox theology distinguishes between ancestral sin and the Western doctrine of original sin, reflecting different anthropological understandings. Original sin, as developed by Augustine and dominant in Western Christianity, suggests that Adam's sin corrupted human nature itself, transmitting guilt and condemnation to all descendants. Humans are born guilty and deserving punishment, requiring baptism for salvation from inherited guilt. Orthodox theology rejects inherited guilt while acknowledging sin's consequences. Ancestral sin refers to the corruption of the human condition following Adam's fall - mortality, suffering, and inclination toward sin - without transferring Adam's personal guilt. Death entered through sin (Romans 5:12), but each person bears responsibility only for their own sins, not Adam's. Orthodox anthropology maintains that humans retain the divine image despite the fall, though it may be obscured. Free will remains intact, allowing cooperation with divine grace in salvation. This perspective emphasizes sin as disease requiring healing rather than crime requiring punishment. Baptism removes the consequences of ancestral sin and mortality's power rather than inherited guilt. The Orthodox understanding supports synergy between divine grace and human response, while Western original sin often emphasizes monergistic salvation. This difference impacts views of infant baptism, salvation, and human capacity for good, reflecting broader divergences between Eastern and Western theological anthropology.",

        # Question 9: Cappadocian Fathers
        "The Cappadocian Fathers - Basil the Great, Gregory of Nazianzus, and Gregory of Nyssa - made foundational contributions to Trinitarian doctrine in the 4th century, establishing terminology and concepts essential to Orthodox theology. They refined understanding of divine persons (hypostases) and essence (ousia), distinguishing between what God is (one essence) and who God is (three persons). Basil the Great established the formula 'one ousia, three hypostases,' clarifying that the Trinity shares one divine nature while maintaining three distinct persons. He emphasized the Spirit's divinity against Pneumatomachi heretics. Gregory of Nazianzus, called 'the Theologian,' provided precise theological language about the Trinity's relations, emphasizing that persons are distinguished by relations (fatherhood, sonship, procession) rather than essential differences. His Five Theological Orations became standard Orthodox Trinitarian theology. Gregory of Nyssa developed the concept of eternal relations and divine infinity, arguing against Eunomian claims that the Son was created. Together, they established that divine persons share identical essence while maintaining real distinctions. Their work grounded the First Council of Constantinople (381) and shaped the Nicene Creed's expansion. The Cappadocians integrated biblical revelation with philosophical precision, creating vocabulary for expressing Trinity's mystery while avoiding both modalism and tritheism. Their influence extends beyond Trinitarian doctrine to mystical theology, ecclesiology, and Christian anthropology.",

        # Question 10: Apophatic Theology
        "Apophatic theology (negative theology) is central to Orthodox theological methodology, emphasizing what cannot be said about God rather than positive assertions. This approach recognizes divine transcendence and the limitations of human language and concepts when describing the infinite. Orthodoxy maintains that God's essence is absolutely unknowable, requiring via negativa to avoid reducing God to creaturely categories. Pseudo-Dionysius the Areopagite (5th-6th century) systematized apophatic theology in works like 'Mystical Theology,' establishing hierarchy of knowing that culminates in unknowing. He argued that the highest divine names must be denied to reach authentic divine knowledge, as God surpasses all human concepts and language. This creates dialectical tension with cataphatic (positive) theology, which affirms divine attributes revealed in Scripture and experience. Orthodox theology balances both approaches: cataphatic theology speaks truly about God's energies and activities, while apophatic theology maintains that God's essence transcends all affirmations. Pseudo-Dionysius influenced major Orthodox theologians including Maximus the Confessor, John of Damascus, and Gregory Palamas. The apophatic tradition emphasizes mystical union through unknowing, intellectual humility before divine mystery, and recognition that theological language points beyond itself. This methodology distinguishes Orthodox theology from Western scholasticism's confidence in rational demonstration and influences Orthodox liturgical language, which often proceeds through negation and paradox.",

        # Question 11: Barlaam vs Hesychasts
        "The theological controversy between Barlaam of Calabria and the hesychasts (1330s-1350s) centered on the nature of divine knowledge and mystical experience. Barlaam, a Calabrian monk and philosopher, attacked hesychast claims of experiencing divine light through the Jesus Prayer, arguing that such experiences were created phenomena, not divine reality. He maintained strict divine transcendence, asserting that no creature could have direct knowledge of God, making hesychast experiences either delusion or prideful presumption. Barlaam emphasized rational theology over mystical experience and criticized hesychast somatic practices as crude materialism. The hesychasts, led by Gregory Palamas, defended their mystical traditions as authentic divine encounter. Palamas developed the essence-energies distinction to explain how genuine divine participation was possible without compromising transcendence. He argued that hesychasts experienced the same uncreated light witnessed at Christ's Transfiguration, making their experiences authentic divine energies rather than created visions. The controversy involved deeper issues about the relationship between philosophy and theology, the role of mystical experience in Christian life, and the nature of divine revelation. Three councils in Constantinople (1341, 1347, 1351) condemned Barlaam and endorsed Palamite theology. This controversy established the essence-energies distinction as official Orthodox doctrine and validated hesychast spirituality as authentically Orthodox. The resolution demonstrated Orthodoxy's commitment to mystical experience as genuine theological source alongside Scripture and Tradition.",

        # Question 12: Perichoresis
        "Perichoresis (circumincession or mutual indwelling) is a fundamental Orthodox doctrine describing the dynamic relationship between divine persons in the Trinity and the two natures in Christ. The term, developed by the Cappadocian Fathers and refined by John of Damascus, indicates that divine persons interpenetrate while maintaining distinct identities. In Trinitarian theology, perichoresis explains how Father, Son, and Holy Spirit share one divine essence while remaining three persons - each person fully indwells the others without confusion or separation. This dynamic unity preserves both divine oneness and personal distinctions, avoiding modalism and tritheism. The Father's monarchy remains intact as source of divine processions, while perichoresis describes the eternal communion of divine life. In Christological application, perichoresis explains the relationship between Christ's divine and human natures, which interpenetrate without mixing or confusion while maintaining their respective properties. This doctrine, formulated to defend Chalcedonian Christology, explains how Christ can be simultaneously fully God and fully human in one person. Divine-human perichoresis in Christ enables human salvation through theosis, as humanity gains access to divine life through the incarnate Logos. Orthodox theology extends perichoretic principles to ecclesiology and spiritual life, understanding Christian community and mystical union as participation in divine perichoresis. This doctrine demonstrates Orthodox emphasis on dynamic relationship and communion rather than static essence, influencing understanding of Trinity, Incarnation, and salvation."
    ]
    for model_name in ["EleutherAI/pythia-6.9b", "tiiuae/falcon-7b", "meta-llama/Llama-3.1-8B-Instruct"]:
        wandb.init(project="Orthodox LLM", config={"model_name": model_name})
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        score_sum, section_sum, sections = 0, 0, ["Basic Theology", "Intermediate Theology", "Advanced Theology"]
        section_scores = [0,0,0,0]
        for i in tqdm(range(12), desc="Generating scores..."):
            question = orthodox_questions[i]
            reference = orthodox_reference_answers[i]
            generated = inference(question, model, tokenizer)
            score = evaluate_orthodox_response(
                question=question,
                generated_response=generated,
                reference_answer=reference
            )
            section_sum += score
            if i > 0 and i % 4 == 0:
                print(f"Section #{i / 4} {sections[int(i / 4)]} Score: {section_sum / 4}")
                section_scores[int(i / 4)] = section_sum / 4
                section_sum = 0
            score_sum += score
        wandb.log({
            "basic_theology_score": section_scores[0],
            "intermediate_theology_score": section_scores[1],
            "advanced_theology_score": section_scores[2],
            "overall_orthodox_score": score_sum / 12
        })
        print(f"Orthodox Theology Score: {score_sum/12}/4.0")
        wandb.finish()
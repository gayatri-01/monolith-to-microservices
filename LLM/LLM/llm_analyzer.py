import openai

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

def analyze_with_llm(comments, business_doc):
    prompt = f"""
    Analyze the following code comments and business use case document to suggest improvements in microservice decomposition:
    1) OrderService
    OrderService.java
    OrderController.java
    OrderRepository.java
    PaymentGateway.java

    2) ProductService
    ProductService.java
    ProductController.java
    ProductRepository.java

    3) CartService
    CartService.java
    CartController.java
    CartRepository.java

    Code Comments:
    {comments}

    Business Use Case Document:
    {business_doc}

    Provide insights on refining microservice boundaries, extracting cross-cutting concerns, and aligning with business domains.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text.strip()


comments = {
    "OrderService.java" : ["Processes a refund for an order using payment gateway","Processes a discount for an order using payment gateway."],
    "PaymentService.java" : ["Processes checkout by calculating the total and calling PaymentGateway"]

}

with open("Ecommerce_Business_UseCase_Excerpt", "r") as f:
    business_doc = f.read()

insights = analyze_with_llm(comments, business_doc)
print(insights)
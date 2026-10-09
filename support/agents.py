import json

from groq import Groq
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status
from .models import Conversation, Message, AgentLog


# Initialize Groq client
client = Groq(api_key=settings.GROQ_API_KEY)

groq_model = settings.GROQ_MODEL


# SUPPORT SYSTEM PROMPT
SUPPORT_SYSTEM_PROMPT = """
You are Maya, a customer support agent at CoolBreeze AC.
You help customers with issues related to their AC orders.

Your responsibilities:
- Always use your tools to gather facts before responding.
- Check order details when the customer mentions their order.
- Check refund history before making any refund-related decisions.
- Be empathetic but honest.

Your personality:
- Friendly and professional.
- Patient even when the customer is angry.
- Clear and concise in your replies.
- No emojis.

Important rules:
- Always check order details first before responding.
- Never approve or deny a refund yourself.
- If a refund decision is needed, tell the customer you are checking with your team.
"""


# MANAGER SYSTEM PROMPT
MANAGER_SYSTEM_PROMPT = """
You are a senior support manager at CoolBreeze AC.
A support agent has escalated a customer case to you for a refund decision.

Your responsibilities:
- Review the case summary carefully.
- Consider the customer's refund history.
- Make a fair refund decision.
- Give a clear reason for your decision.

Your decision options:
- Approve refund: if the case is genuine and within policy.
- Deny refund: if the case is suspicious or outside policy.
- Escalate to risk team: if you suspect fraud.

Important rules:
- Be fair but firm.
- Base decisions on facts, not emotions.
- Always give a specific reason for your decision.
- Keep your response concise and professional.
"""


# SUPPORT TOOLS: Groq-compatible function schemas
SUPPORTS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Fetch order details including status, carrier, tracking "
                "number, and days since the order was placed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID to look up."
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_history",
            "description": (
                "Get the complete refund history for a user. "
                "Use this before making refund-related decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user ID to check."
                    }
                },
                "required": ["user_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_delivery_status",
            "description": (
                "Check delivery status using the tracking number and carrier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "The shipment tracking number."
                    },
                    "carrier": {
                        "type": "string",
                        "description": "The carrier name."
                    }
                },
                "required": ["tracking_number", "carrier"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_manager",
            "description": (
                "Escalate a customer case to the manager for a refund "
                "decision. Use this when the customer requests a refund."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_summary": {
                        "type": "string",
                        "description": (
                            "Case summary including order details, refund "
                            "history, and customer complaint."
                        )
                    }
                },
                "required": ["case_summary"],
                "additionalProperties": False
            }
        }
    }
]


# Execute tools: bridge between Groq and Python functions
def execute_tools(tool_name, tool_input):

    if tool_name == "get_order_details":
        return get_order_details(tool_input["order_id"])

    elif tool_name == "get_refund_history":
        return get_refund_history(tool_input["user_id"])

    elif tool_name == "check_delivery_status":
        return check_delivery_status(
            tool_input["tracking_number"],
            tool_input["carrier"]
        )

    elif tool_name == "escalate_to_manager":
        case_summary = tool_input["case_summary"]
        return run_manager_agents(case_summary)

    return {"error": f"Unknown tool: {tool_name}"}


# Support Agent: Groq tool-calling loop
def run_support_agent(user_message, conversation_id, order_id, user_id):

    conv = Conversation.objects.get(id=conversation_id)

    conversation_messages = []

    for msg in conv.messages.order_by("created_at"):
        conversation_messages.append({
            "role": msg.role,
            "content": msg.content
        })

    messages = [
        {
            "role": "system",
            "content": SUPPORT_SYSTEM_PROMPT
            + f"\n\nContext: Order #{order_id}, User ID: {user_id}."
        },
        *conversation_messages
    ]

    # Add the current user message if it is not already in history.
    # This assumes the calling view has not saved it already.
    if not conversation_messages or (
        conversation_messages[-1]["role"] != "user"
        or conversation_messages[-1]["content"] != user_message
    ):
        messages.append({
            "role": "user",
            "content": user_message
        })

    # Prevent an endless tool-calling loop.
    max_iterations = 10

    for _ in range(max_iterations):

        response = client.chat.completions.create(
            model=groq_model,
            max_tokens=1024,
            messages=messages,
            tools=SUPPORTS_TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # If Groq does not request a tool, return its final response.
        if not assistant_message.tool_calls:
            return assistant_message.content or (
                "I'm sorry, I couldn't generate a response. Please try again."
            )

        # Append the assistant's tool calls to the conversation.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        })

        # Execute requested tools and return their results to Groq.
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            try:
                tool_input = json.loads(
                    tool_call.function.arguments
                )

                tool_result = execute_tools(
                    tool_name,
                    tool_input
                )

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                tool_result = {
                    "error": f"Invalid tool arguments: {str(exc)}"
                }

            except Exception:
                tool_result = {
                    "error": "The requested tool could not be executed."
                }

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    tool_result,
                    default=str
                )
            })

    return "I'm sorry, I couldn't complete the request. Please try again."


# Manager Agent: Groq-compatible implementation
def run_manager_agents(case_summary):

    manager_messages = [
        {
            "role": "system",
            "content": MANAGER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": case_summary
        }
    ]

    # The manager makes a decision using the supplied case facts.
    # No tools are required for this decision.
    response = client.chat.completions.create(
        model=groq_model,
        max_tokens=1024,
        messages=manager_messages
    )

    return response.choices[0].message.content or (
        "The manager could not generate a decision. Please escalate "
        "this case for manual review."
    )

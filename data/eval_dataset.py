# data/eval_dataset.py

EVAL_QUESTIONS = [
    # Factual (Directly from text)
    "What is the definition of workflow automation?",
    "What are the three key components of workflow automation?",
    "How does n8n describe its approach to coding and flexibility compared to other tools?",
    "What is the starting price for n8n's managed cloud plans?",
    "What visual indicator is used to identify a trigger node in n8n?",
    "Which built-in node is used to perform HTTP requests to interact with external APIs?",
    "What format does n8n use to output payload data from nodes?",
    "What is the difference between relative referencing and absolute referencing in n8n?",
    "What is the primary function of the 'Edit Fields (Set)' node?",
    "What are the two primary execution modes available in n8n?",
    
    # Multi-section / Conceptual
    "If I wanted to compare the cost and integrations of Zapier, Make.com, and n8n, what are the key differences?",
    "Describe the three main types of nodes involved in an ETL process within n8n.",
    "What are the main advantages and disadvantages of choosing self-hosting over n8n Cloud?",
    "Explain how the 'Code Node' can be used for advanced data manipulations, and give an example of its expected output format.",
    "If a workflow encounters an error, what are two different ways a user can configure n8n to handle it?",
    "How does n8n differentiate between the $json variable and the $env variable?",
    "Describe the four types of AI Agents mentioned in the guide.",
    "What are the key elements of a good prompt when interacting with AI agents?",
    "How did the organization SanctifAI utilize n8n in their workflow, and what was the outcome?",
    "What are the four implementation steps recommended for building practical AI agent workflows?",
    
    # Specific / Numerical / Edge Cases
    "How many pre-built integrations does n8n offer?",
    "If Zapier charges $19.99 per month, how much does Make.com's paid plan start at?",
    "When using expressions in n8n, what syntax would you use to retrieve the 'name' field inside a 'customer' object from the current node's data?",
    "If I need to restrict the number of items passed to a subsequent node, which specific 'no-code' node should I use?",
    "When using temperature control in an AI prompt, what is the difference between using a temperature of 0.2 versus 0.8?"
]

EVAL_GROUND_TRUTHS = [
    # Factual (Directly from text)
    "Workflow automation is the process of using technology to perform tasks or processes without manual intervention. It involves setting up rules and triggers that automate repetitive or predictable actions, ensuring tasks happen consistently and efficiently.",
    "The three key components of workflow automation are: 1) Trigger Events, which start a workflow; 2) Actions, which are the tasks performed in response to a trigger; and 3) Conditions, which ensure actions only happen when certain criteria are met.",
    "n8n is a low-code, node-based workflow automation tool known for its flexibility. It is a 'fair-code' platform that allows unparalleled customization, supporting complex workflows with conditional logic, loops, and error handling. Developers can also add custom JavaScript via Code Nodes for enhanced versatility and have the option to self-host the environment.",
    "n8n offers managed cloud plans starting at $20 per month, which include hosting and additional features like enterprise solutions.",
    "A trigger node is identified by an orange lightning bolt.",
    "The HTTP Request Node is used to perform HTTP requests to interact with external APIs, supporting methods like GET, POST, PUT, and DELETE.",
    "In n8n, nodes output a payload of data represented as a JSON array of objects.",
    "Relative referencing uses $json to access the data of the immediately previous node (or the current node where the expression is written). Absolute referencing uses $node[\"NodeName\"].json to access data from a specific previous node in the workflow.",
    "The Edit Fields (Set) Node defines and manipulates data fields within a workflow. It allows users to add predefined static values (constants) for use in later nodes, or to map and transform dynamic incoming data to new fields.",
    "The two primary execution modes are Manual Execution, which is used during the development phase to test workflows and allows step-by-step observation, and Production Execution, where activated workflows run automatically in live operations based on defined triggers.",
    
    # Multi-section / Conceptual
    "Zapier offers over 5,000 app integrations with plans starting at $19.99 per month based on tasks. Make.com supports over 1,500 integrations with plans starting at $9 per month based on operations. n8n offers over 300 pre-built integrations (plus custom community nodes), is free to self-host for complete customization, and offers managed cloud plans starting at $20 per month.",
    "The three main types of nodes are: 1) Trigger Nodes, which initiate workflows based on events or schedules; 2) Core Nodes (the 'Transform' step in ETL), which process and format data using custom logic; and 3) Action Nodes (the 'Extract/Load' step in ETL), which perform final specific actions like sending a message or uploading a file.",
    "n8n Cloud is managed by the n8n team and requires minimal setup, making it ideal for beginners, but it operates on a paid subscription. Self-hosting is free and provides full control over data, infrastructure, and security compliance, but it requires technical expertise to set up and maintain using methods like Docker, Node.js, or k8s.",
    "The Code Node executes custom JavaScript for complex data transformations that standard nodes cannot handle, such as date conversions, array filtering, or calculations. The output must always be a valid n8n item format, which is an array of objects (e.g., return [{ key1: \"value1\" }];) or an empty array (e.g., return [];).",
    "A user can set up an Error Workflow using the 'Error Trigger' node, which automatically activates a separate workflow to send notifications (like emails or Slack messages) when an error occurs. Alternatively, a user can enable the 'Continue On Fail' option in a node's settings, allowing the node to continue executing and the workflow to proceed with alternative actions.",
    "The $json helper variable represents the data of the current node and is commonly used to access input data within the node where the expression is written. The $env variable enables access to environment variables set in the n8n instance, which is useful for managing configuration values that vary between environments, such as development, staging, or production.",
    "The four types of AI Agents are: 1) Reactive Agents, which respond to stimuli without internal states or historical context; 2) Goal-Based Agents, which act to achieve specific objectives using planning; 3) Utility-Based Agents, which evaluate actions based on a utility function to maximize performance; and 4) Learning Agents, which improve performance over time by learning from experiences.",
    "The key elements of a good prompt are: 1) Clarity, being explicit about what you want the AI to do; 2) Context, providing background information or assigning a role; 3) Constraints, specifying rules or limitations for the output; and 4) Examples (Multi-shot Prompting), providing one or more examples to guide the AI on the desired format, tone, or structure.",
    "SanctifAI utilized n8n to design workflows that incorporate human inputs at critical decision points, combining AI processing with human judgment to enhance accuracy and reliability. The outcome was a 3x increase in workflow development speed compared to traditional methods.",
    "The four implementation steps for AI agents are: 1) Define Objectives, clearly identifying the tasks and goals; 2) Select AI Tools, choosing appropriate AI models or services; 3) Design Workflow, creating n8n workflows that integrate the selected AI tools; and 4) Monitor and Improve, continuously monitoring performance to make necessary adjustments.",
    
    # Specific / Numerical / Edge Cases
    "n8n offers over 300 pre-built integrations.",
    "Make.com's paid plan starts at $9 per month.",
    "You would use the syntax {{$json[\"customer\"][\"name\"]}} or dot notation such as {{$json.customer.name}}.",
    "You should use the Limit node to restrict the number of items passed to subsequent nodes.",
    "A lower temperature (e.g., 0.2) is used to generate precise, factual answers, while a higher temperature (e.g., 0.8) is used to generate creative outputs, such as writing a fictional story."
]

import re

def clean_ground_truths(truths):
    """Remove citation markers like [cite: 197, 198] from ground truths."""
    return [re.sub(r'\[cite:.*?\]', '', truth).strip() for truth in truths]

EVAL_GROUND_TRUTHS_CLEAN = clean_ground_truths(EVAL_GROUND_TRUTHS)
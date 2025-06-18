###### Trabajo Final del Grado en Ciencia de Datos.
###### Final thesis work, BSc. in Data Science.
###### Universitat Politècnica de València.

## Abstract

The emergence of large language models has led to the creation of many AI assistants. While a lot of
focus has been placed on massive models, less attention is paid to small models, which are often
significantly cheaper to run and still have good analytical abilities. This project builds and evaluates a
ReAct-based assistant with small language models. To do this, an intelligent breakdown of tasks is done
into small reasoning steps, and great care is taken to adequately retrieve the decisions of the model.


Models are aided by two custom-made tools, to do web search and coding, allowing for an agentic
behaviour. The system is also designed for a great degree of personalization, leading to a better user
experience. To do this, a new approach is taken to extract, store, and adequately employ user
information. A final evaluation of the system is carried out, featuring the performance comparison of
different small models and one large model.


The solution is custom built and non-dependent on third party abstraction libraries to interact with
the language models (such as langchain). This provides a more efficient use of the calls and more
customization over the use of the program. It also represents a better opportunity to learn from the
inner workings and details of LLM systems.


During the development, interesting issues arose that point to new challenges specific to LLM
applications. These involved the difficulty of parsing answers from small-sized models, managing and
versioning the system prompts, and the appearance of a new type of bugs related to prompts, which
I call “prompt bugs”, within the reasoning system. The report explores these issues and reflects on the
lessons learned together with recommendations for better design of LLM applications.

###### Key words:
LLM, Large Language Models, Agentic systems, Retrieval Augmented Generation, RAG, AI,
Small Language Models, Prompt Engineering.


## Goals

> The goal of this project is to build an agent-based, personalized AI assistant powered by small language models, and through this, evaluate the capabilities of SLMs to power such a system.


This project is focused on the learning outcomes of the experience and is not meant to achieve the
performance levels of mainstream AI assistants, which are built by teams of top researchers in big AI
companies. However, it tries to answer questions such as: can we reliably use SLMs as the main
processing power of AI systems? What are the main challenges when powering applications with
SLMs? Can SLMs effectively choose and call the appropriate tool in an agentic setup? How can AI
assistants be improved through personalization?
These general objectives can be specified into a more detailed breakdown:
- Develop a personalized AI assistant capable of answering moderately difficult queries. The
program must be able to store current conversation information and leverage its reasoning
system through sequential and logical calls to the SLM to arrive at a final answer. It must also
automatically identify, store, and adequately retrieve and use relevant user information to help
to better tailor its answers to each user.
- Allow for a separation of difficulty levels of intermediate language model processing steps. The
system must be able to handle two different qualities of models, to leave the most complicated
steps to the best model and use the lower quality and cheaper model to handle less critical
steps. This will make the system more cost efficient when deployed.
- Identify and develop the necessary tools to power the assistant. Language models, especially
small ones, need external tools to be able to handle nontrivial queries. Use the Reason-Act
framework to guide the reasoning steps and select the adequate tool. To avoid
overrunning the SLM capabilities, special care must be taken to separate all internal tasks into
very simple actionable steps.
- Create a mechanism to correctly identify the call of tools and their input. Parse the raw model
output to identify, within the ReAct steps, the correct tool calls and their associated input.
Then, run the tools with basic troubleshooting and retry mechanisms and return their result.
The parser should also correctly identify when the model has reached the final answer.
- Design a scalable architecture that easily allows to integrate new tools and models in the
future. Scalability will help the system to evolve, by improving the current tools and adding
new technology as it becomes available. It also simplifies troubleshooting and code
maintenance.
12
- Create a traceable and organized system to store the prompts. Collect all used system prompts
into a single location and make the relevant loading from the code. This will improve
maintainability by forbidding to store prompts in random, maybe hidden places in the code.
- Perform a thorough evaluation of the system. Include several candidate models to try to
identify the best performing ones. Also, a comparison should be made against a reference
large language model.

## Solution

Note: overview. For detailed explanation, reasonings, and architecture, refer to the project document.

![Todos los diagramas_page-0001](https://github.com/user-attachments/assets/51d5eab0-2561-4d86-93d4-37039939ca09)
![Todos los diagramas_page-0002](https://github.com/user-attachments/assets/5fc35ae2-b77d-4e97-b2d4-2e3c76f30727)


#### Ejecutar el código

Con el comando "python -m streamlit run app.py"

# import dependencies
import streamlit as st
# for app framework and Validations
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory 
# For Search and Memory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
# For autoGPT model
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
# For search 
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
#
import faiss
from contextlib import redirect_stdout
import pandas as pd
import json
import os
import re
import time
import csv

#

#
# CLASSES AND FUNCTIONS
class StreamlitOutput:
	def __init__(self, stapp):
		self.buffer = ""
		self.stapp = stapp

	def write(self, data):
		self.buffer += data
		# check if data is json
		if 'Result: ' not in data:
			data = data.strip()
			if(len(data) > 2):	
				if(data[0] == '{' and data[-1] == '}'):
					try:
						plan = json.loads(data)["thoughts"]["plan"]
						self.stapp.write(plan)
						self.stapp.json(data, expanded=False)
					except:
						self.stapp.json(data, expanded=False)
				else:
					self.stapp.write(data)
		# if 'File written successfully to ' in data:
		# 	match = re.search(r'File written successfully to ([\w\.-]+)\.', data)
		# 	if match:
		# 		file_path = match.group(1)
		# 		if 'file_paths' not in st.session_state:
		# 			st.session_state.file_paths = []
		# 		if file_path not in st.session_state.file_paths:
		# 			st.session_state.file_paths.append(file_path)

def load_files():
	# If 'file_paths' exists in the session state, iterate through each file path
	if 'file_paths' in st.session_state:
		for file_path in st.session_state.file_paths:
			# Check if the file still exists
			if os.path.exists(file_path):
				# If the file exists, create a download button for it
				with open(file_path, "rb") as file:
					file_data = file.read()
				st.download_button(
					label=f"Download {os.path.basename(file_path)}",
					data=file_data,
					file_name=os.path.basename(file_path),
					mime="text/plain",
					key=file_path  # Use the file path as the key
				)
			else:
				# If the file doesn't exist, remove it from 'file_paths'
				st.session_state.file_paths.remove(file_path)
	else:
			# If 'file_paths' doesn't exist in the session state, create it
			st.session_state.file_paths = []


#
#
# Agent parameters
st.sidebar.title("AutoGPT Parameters")
ai_name = st.sidebar.text_input(
    "AI Name",
    value="StarGPT",
    help="Enter the name of your AI agent.",
)
ai_role = st.sidebar.text_input(
    "AI Role",
    value="Intern at a top notch Venture capital firm",# at a top notch Venture capital firm which has invested in a total of 80 companies to date and the portfolio value exceeds $500 million. Your work will help the firm understand the start-up landscape better.
    help="Enter the role of your AI agent.",
)
MAX_COMPANIES = st.sidebar.number_input(
		"Max Companies",
		value=2,
		help="Enter the maximum number of companies you want to search for.",
		min_value=1,
		max_value=20,
		step=1,
	)

open_ai_key = st.sidebar.text_input(
    "OpenAI API Key",
    value="",
    help="Enter your OpenAI API key.",
)
serp_api_key = st.sidebar.text_input(
    "SerpAPI API Key",
    value="",
    help="Enter your SerpAPI API key.",
)

#
#
# Setup keys
os.environ['OPENAI_API_KEY'] = open_ai_key#
os.environ['SERPAPI_API_KEY'] = serp_api_key#

if(len(os.environ['OPENAI_API_KEY']) == 0):
	st.error("Please enter your OpenAI API key in the sidebar.")
	st.stop()
if(len(os.environ['SERPAPI_API_KEY']) == 0):
	st.error("Please enter your SerpAPI API key in the sidebar.")
	st.stop()

#
#
# Setup tools and memory
search = SerpAPIWrapper(serpapi_api_key=os.environ['SERPAPI_API_KEY'])
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Setup AutoGPT
top_agent = AutoGPT.from_llm_and_tools(
    ai_name=ai_name,
    ai_role=ai_role,
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
top_agent.chain.verbose = False


# Setup output stream to streamlit
output = StreamlitOutput(st)

# Setup streamlit app
# App framework
st.title('🚀💡 Startup evaluator')

if 'user_input' not in st.session_state:
	st.session_state.user_input = ""
st.session_state.user_input = st.text_input('Enter startup category and location here. Eg: "Fintech startups in India"', st.session_state.user_input)

if st.button('Run'):
	if st.session_state.user_input:
		with redirect_stdout(output):
			#
			# check if prompt contains "startup in"
			prompt = st.session_state.user_input
			if "startups in" in prompt:
				# split prompt into category and location
				category, location = prompt.split("startups in")
				# Trim whitespace in category and location
				category = category.strip()
				location = location.strip()
				#print(category)
				#print(location)
				#
				st.info('Verifying inputs...')
				# Prompt templates for verifying category and location
				category_template = PromptTemplate(
						input_variables = ['category'],
						template='check if {category} is a valid startup category. Start your answer with "Yes," or "No," and provide a reason'
				)
				location_template = PromptTemplate(
						input_variables = ['location'],
						template='check if {location} is a valid location. Start your answer with "Yes," or "No," and provide a reason'
				)
				# Memory
				llm = OpenAI(temperature=0.0)
				category_memory = ConversationBufferMemory(input_key='category', memory_key='chat_history') 
				location_memory = ConversationBufferMemory(input_key='location', memory_key='chat_history')
				# Llms
				category_chain = LLMChain(llm=llm, prompt=category_template, verbose=False, output_key='category', memory=category_memory)
				location_chain = LLMChain(llm=llm, prompt=location_template, verbose=False, output_key='location', memory=location_memory)
				#
				category_output = category_chain.run(category=category)
				location_output = location_chain.run(location=location)
				# Show stuff to the screen
				st.write(category_output)
				st.write(location_output)
				# check if category_output and location_output starts with "Yes,"
				check_category = category_output.split(",")[0].strip()
				check_location = location_output.split(",")[0].strip()
				#print(check_category)
				#print(check_location)
				if check_category.__eq__("Yes") and check_location.__eq__("Yes"):
					if os.path.isfile('potential_companies.txt'):
						os.remove('potential_companies.txt')
					if os.path.isfile('company_data.tsv'):
						os.remove('company_data.tsv')
					# Prompt template for identifying top 10 startups
					#print('VERIFIED')
					st.write('Now identifying top '+ str(MAX_COMPANIES) +' ' + prompt)
					# Run the agent
					# use append_to_file instead of write_to_file to add more to the list.
					result = top_agent.run(['Goal 1: Search for '+ str(MAX_COMPANIES)+' '+category + ' startups based in '+location+ '.\nGoal 2: Make a numbered list of the companies and make sure that you have atleast '+str(MAX_COMPANIES)+' companies in the list.\nGoal 3: Save the list as potential_companies.txt \nGoal 4: Save the list as potential_companies.txt \nGoal : Use "finish" command after writing text file.'])
					#
					#if 'file_paths' not in st.session_state:
					#	st.session_state.file_paths = []
					#st.session_state.file_paths.append('potential_companies.txt')
					#
					# Researching each company
					st.info('Now researching each company...')
					time.sleep(10)
					# load file potential_companies.txt
					with open('potential_companies.txt', 'r') as f:
							time.sleep(10)
							potential_companies = f.read().splitlines()
							if(len(potential_companies[0]) == 0):
								potential_companies = potential_companies[0].split('\n')
							# check number of lines in potential_companies.txt
							numlines = len(potential_companies)
							if(numlines >= int(MAX_COMPANIES)):
								# remove additional companies
								for i in range(int(MAX_COMPANIES), numlines):
									potential_companies.pop()
							# save potential_companies.txt
							with open('potential_companies.txt', 'w') as f:
								for company in potential_companies:
									f.write("%s\n" % company)	
							# loop over potential_companies list
							for company in potential_companies:
								company_name = company.split('. ')[1]
								st.subheader(company_name+':')
								#
								if os.path.isfile(company_name+'_company_info.tsv'):
									os.remove(company_name+'_company_info.tsv')
								# Prompt for searching each company
								company_template = PromptTemplate(
										input_variables = ['company_name'],
										template='search {company_name} and write a short description about the company'
								)
								llm = OpenAI(temperature=0.0)
								company_memory = ConversationBufferMemory(input_key='company_name', memory_key='chat_history')
								company_chain = LLMChain(llm=llm, prompt=company_template, verbose=False, output_key='company_name', memory=company_memory)
								company_output = company_chain.run(company_name=company_name)
								st.write(company_output)
								#
								company_agent = AutoGPT.from_llm_and_tools(
										ai_name=ai_name,
										ai_role=ai_role,
										tools=tools,
										llm=ChatOpenAI(temperature=0),
										memory=vectorstore.as_retriever()
								)
								company_agent.chain.verbose = False
								result = company_agent.run(['Goal 1: I have identified '+company_name+' which shows investment potential. Use "search" command to learn about '+company_name+'.\nGoal 2: Convert your research into a tsv format and write to file finalfile.tsv with the headers: Name\tCompany website\tServices offered\tIndustry\tMarket size\tFundraising round (if any)\tExisting investors (if any)\tEmployee size.\nGoal 3: Here is an example. Research format is: Afterpay - https://www.afterpay.com/ - Buy now, pay later platform - $200B domestic retail industry - ASX-listed - Not specified - 700 employees\nAfterpay is a leading buy now, pay later platform that has seen significant growth in recent years. The company raised $300 million in debt funding in 2020 to support its expansion plans.Afterpay is a global fintech company that offers a buy now, pay later service for consumers. The company has over 16 million customers and over 98,000 merchants worldwide. Afterpay has raised over $1.5 billion in funding to date and has a valuation of over $30 billion. Existing investors include Tencent, Sequoia Capital, and Coatue Management. tsv format is: Name\tWebsite\tServices offered\tIndustry\tMarket Size\tFundraising round\tExisting investors\nAfterpay\thttps://www.afterpay.com/\tBuy now, pay later platform\tdomestic retail industry\t$200B domestic retail industry\t$300 million in debt funding in 2020\tTencent, Sequoia Capital, and Coatue Management\t700-1000"\nGoal 4: save the tsv formatted data to finalfile.tsv and make sure 1st line is header and second line is the required data. No additional characters allowed.\nGoal 5: Use "finish" command after writing to tsv file.'])
								#
								time.sleep(5)
								if os.path.isfile('finalfile.tsv'):
									os.rename('finalfile.tsv', company_name+'_company_info.tsv')
								else:
									st.error("Failed to save file "+company_name+"_company_info.tsv")
								#
					#
					st.info('Now filling missing data... (if any)')
					# Fill missing data
					with open('potential_companies.txt', 'r') as f:
						potential_companies = f.read().splitlines()
						if(len(potential_companies[0]) == 0):
							potential_companies = potential_companies[0].split('\n')
						# loop over potential_companies list and update company_info.tsv
						for company in potential_companies:
							company_name = company.split('. ')[1]
							print(company)
							if os.path.isfile(company_name+'_company_info.tsv'):
								#
								# read and update each company_info.tsv
								with open(company_name+'_company_info.tsv', 'r') as f:
									all_lines = f.readlines()
									numlines = len(all_lines)
									if(numlines == 1):
										all_lines = all_lines[0].split('\n')
										numlines = len(all_lines)
									if(numlines >= 2):
										# Read single row
										header_row = all_lines[0].split('\t')
										# Read remaining row
										company_row = all_lines[1].split('\t')
										# index of 'Market size', 'Fundraising round (if any)', 'Existing investors (if any)' and 'Employee size'
										industry_index = -1
										market_size_index = -1
										fundraising_round_index = -1
										existing_investors_index = -1
										employee_size_index = -1
										for i in range(len(header_row)):
											#print(header_row[i])
											#make header_row lowercase
											hr = header_row[i].lower()
											if 'industry' in hr:
												industry_index = i
											if 'market size' in hr:
												market_size_index = i
											elif 'fundraising round' in hr:
												fundraising_round_index = i
											elif 'existing investors' in hr:
												existing_investors_index = i
											elif 'employee size' in hr:
												employee_size_index = i
										# Check if index is not -1
										if industry_index != -1 and market_size_index != -1 and fundraising_round_index != -1 and existing_investors_index != -1 and employee_size_index != -1:
											# Deep dive into 'Market size', 'Fundraising round (if any)', 'Existing investors (if any)' and 'Employee size'
											tools = load_tools(['serpapi'])
											# Setup agent
											this_company_agent = initialize_agent(tools=tools, llm=OpenAI(temperature=0), agent='zero-shot-react-description', verbose=False)
											current_industry = company_row[industry_index]
											#print(company_row[market_size_index])
											# print(company_row[fundraising_round_index])
											# print(company_row[existing_investors_index])
											# print(company_row[employee_size_index])
											if 'not ' in company_row[market_size_index].lower():
												result = this_company_agent.run('What is the Market size of ' + current_industry + ' in '+ location+'? in billion dollars only.')#'Goal1: Use "search" command to specifically get the Market size of '+company_name+'.\nGoal2: The answer should be very consise and to the point. Do not form a sentence.\nGoal3: Do not write to any file. Just print the answer.\nGoal4: When you know the answer use "finish" command.'
												company_row[market_size_index] = result
											time.sleep(5)
											if 'not ' in company_row[fundraising_round_index].lower():
												result = this_company_agent.run('Find the total Fundraising round of '+company_name + '? in million dollars only.')
												company_row[fundraising_round_index] = result
											time.sleep(5)
											if 'not ' in company_row[existing_investors_index].lower():
												result = this_company_agent.run('Who are the existing investors of '+company_name + '? Give me investor names only.')
												company_row[existing_investors_index] = result
											time.sleep(5)
											if 'not ' in company_row[employee_size_index].lower():
												result = this_company_agent.run('What is the total employee size of '+company_name + '? the number or a range only.')
												company_row[employee_size_index] = result
											time.sleep(10)
											# Write to csv file
											with open(company_name+'_company_info.tsv', 'w') as f:
												# convert header_row into string
												header_row = '\t'.join(header_row)
												# convert company_row into string
												company_row = '\t'.join(company_row)
												# write header_row and company_row
												f.write(header_row)
												f.write('\n')
												f.write(company_row)
											#
										else:
											st.error("Failed to find index of 'Market size', 'Fundraising round (if any)', 'Existing investors (if any)' and 'Employee size'")
											print(header_row)
								#
						# loop over potential_companies list and merge into company_data.tsv
						company_header = []
						company_data = []
						# Load company_info.tsv
						for company in potential_companies:
							company_name = company.split('. ')[1]
							#print(company_name)
							if os.path.isfile(company_name+'_company_info.tsv'):
								# read and load each company_info.tsv
								with open(company_name+'_company_info.tsv', 'r') as f:
									all_lines = f.readlines()
									numlines = len(all_lines)
									if(numlines == 1):
										all_lines = all_lines[0].split('\n')
										numlines = len(all_lines)
									if(numlines >= 2):
										# Read single row
										header_row = all_lines[0]
										if(len(company_header) == 0):
											company_header = header_row
										# Read remaining row
										if(all_lines[1] == '\n'):
											company_row = all_lines[2]
										else:
											company_row = all_lines[1]
										company_data.append(company_row)
									# Delete company_info.tsv
									#os.remove(company_name+'_company_info.tsv')
						#print(company_header)
						#print(company_data)
						company_header = company_header.split('\t')
						for i in range(len(company_header)):
							company_header[i] = company_header[i].strip()
							company_header[i] = company_header[i].replace(',', '')
						company_header = '\t'.join(company_header)
						# loop over company_data and remove any commas
						for i in range(len(company_data)):
							company_row = company_data[i]
							company_row = company_row.split('\t')
							for j in range(len(company_row)):
								company_row[j] = company_row[j].strip()
								company_row[j] = company_row[j].replace(',', '')
							company_data[i] = '\t'.join(company_row)
						# write to company_data.csv
						with open('company_data.csv', 'w', newline='') as f:
							writer = csv.writer(f)
							company_header = company_header.split('\t')
							writer.writerow(company_header)
							for row in company_data:
								row = row.split('\t')
								writer.writerow(row)
						#
						# Add company_data to streamlit
						if 'file_paths' not in st.session_state:
							st.session_state.file_paths = []
						st.session_state.file_paths.append('company_data.csv')
						#
						df = pd.read_csv('company_data.csv')
						st.table(df)
						#
						# ENDS HERE
						#
				else:
					st.write("Please enter a valid startup category and location")
			else:
				st.write("Please enter a valid prompt")
			#
		st.session_state.model_output = output.buffer
		st.session_state.result = result
		load_files()
		st.session_state.files_loaded = True
		# If any of the download buttons was clicked, re-render them
		if 'file_paths' in st.session_state:
			for index, file_path in enumerate(st.session_state.file_paths):
				widget_key = f"{file_path}_{index}"  # Use the unique widget key
				if st.session_state.get(widget_key, False):  # Check if the button was clicked
					load_files()
		# Display the model output if it's in the session state
		if 'model_output' in st.session_state:
			#st.write(f"Result: {st.session_state.result}")
			expander = st.expander("Model Output")
			expander.text(st.session_state.model_output)

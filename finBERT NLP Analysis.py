#!/usr/bin/env python
# coding: utf-8

# ### Packages

# In[1]:


import requests
from bs4 import BeautifulSoup
import csv
get_ipython().system('pip install ipython')
get_ipython().system('pip install nbconvert')


# ### Out Look

# In[2]:


def Get_Words(input):
    noise = ["\xa0\r\n\xa0", "\xa0", '\'', "\n", "           "," ' ", '\"',"\t\t\t\t", "By Janet Henry, Global Chief Economist, HSBCVideoGlobal Economics QuarterlyGlobal growth was more resilient than many feared through the third quarter of 2022. ", "JHVEPhoto/iStock Editorial via Getty ImagesIn one of the most difficult macroeconomic environments in a decade, Capital One Financial Corporation () does not appear to be a good stock to buy. ", "\t\t\t", "Download the Schwab app from iTunesCross currents continue to rock the economic boat, even though we believe a brighter year is on the horizon.", "|  |  |  |  |  |  |  |  | ||", "Watch CBS News By Aimee Picchi         / MoneyWatch        Investors , with the S&P 500 tumbling 19% as the Federal Reserve cranked interest rates higher to smother inflation. ", "Please .Corporations & InstitutionsInstitutional InvestorsAdvisors & Advisory FirmsIndividuals & Portfolio AdvisorsAs a global leader, we deliver strategic advice and solutions, including capital raising, risk management, and trade finance services to corporations, institutions and governments.Serving the worlds largest corporate clients and institutional investors, we support the entire investment cycle with market-leading research, analytics, execution and investor services.We are a leader in investment management, dedicating to creating a strategic advantage for institutions by connecting clients with J.P.Morgan investment professionals globally. Our financial advisors create solutions addressing strategic investment approaches, professional portfolio management and a broad range of wealth management services. InsightsResearchInsights to empower better decisions. Leverages cutting-edge technologies and innovative tools to bring clients industry-leading analysis and investment advice. NewsroomMedia CenterInvestor RelationsThe latest news and announcements. For company information and brand assets for editorial use.: The latest news and announcements.: About usCorporate ResponsibilityTechnology at our FirmEvents & ConferencesIn a fast-moving and increasingly complex global economy, our success depends on how faithfully we adhere to our core principles: delivering exceptional client service; acting with integrity and responsibility; and supporting the growth of our employees. J.P.Morgan is a global leader in financial services, offering solutions to the worlds most important corporations, governments and institutions in more than 100 countries. ", "Quotes displayed in real-time or delayed by at least 15 minutes. Market data provided by. Powered and implemented by. . Mutual Fund and ETF data provided by.          This material may not be published, broadcast, rewritten, or redistributed. ©2023 FOX News Network, LLC. All rights reserved.  - "]
    output = ""
    text = input.find_all(text = True)
    for x in text:
        if x.parent.name == "p":
            output += "{}".format(x)
    output = str(output).strip()
    for x in noise:
        output = output.replace(x, "")
    return output

def Get_Text(site):
    bank_page = requests.get(site)
    page_info = BeautifulSoup(bank_page.content, "html.parser")
    return Get_Words(page_info)


# ### Aggregrate Sentiment

# In[3]:


def aggragated_sentiments(array):
    scores = 0
    total_negative = 0
    total_positive = 0
    total_neutral = 0
    for x in range(0,len(array)):
        if (array[x] == "Negative"):
            total_negative += 1
        if (array[x] == "Positive"):
            total_positive += 1
        if (array[x] == "Neutral"):
            total_neutral += 1
        if type(array[x]) == float:
            scores = scores + array[x]
    score = scores/len(array)
    Data = {
        "Average Score": score,
        "Total Negative's" : total_negative, 
        "Total Positive's" : total_positive,
        "Total Neutral's" : total_neutral
    }
    return Data


# In[4]:


def total_sentiment(array):
    scores = 0
    total_negative = 0
    total_positive = 0
    total_neutral = 0
    for x in array:
        scores += x["Average Score"]
        total_negative += x["Total Negative's"]
        total_positive += x["Total Positive's"]
        total_neutral += x["Total Neutral's"]
    scores = scores/(len(array))
    sentiment = ""
    if scores < 0.5:
        sentiment = "Negative Aggregrate Outlook"
    if scores >= 0.5 and scores <= 0.6:
        sentiment = "Neutral Aggregrate Outlook"
    if scores > 0.6:
        sentiment = "Positive Aggregrate Outlook"
    Total_Data = {
    
    "Total Average Score": scores,
    "Total Negative's" : total_negative, 
    "Total Positive's" : total_positive,
    "Total Neutral's" : total_neutral,
    "Average Sentiment" : sentiment, 
    "Banks Considered": ["Goldman Sachs", "J.P. Morgan", "Bank of America Corporation" , "Citigroup", "Wells Fargo", "Morgan Stanley", 
        "Charles Schwab", "US Bank Corporation", "TD Bank", "Capitol One Bank", "State Street Corp", "HSBC Bank USA"]
    }
    
    return Total_Data


# ### finBERT NLP Model

# In[5]:


get_ipython().system('pip install transformers')

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels = 3)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

NLP = pipeline("sentiment-analysis", model=finbert, tokenizer = tokenizer)

def sentiment_analysis(text):
    NLP = pipeline("sentiment-analysis", model=finbert, tokenizer = tokenizer)
    results = NLP(text)
    Label = results[0]["label"]
    Score = results[0]["score"]
    return [Label, Score]


# ### Sizeable Data

# In[6]:


def get_sizeable_sentences(text):
    sentences = []
    current_sentence = ""
    letter = ""
    x = 0
    for x in range(0, len(text) - 5):
        current_sentence += text[x]
        if (text[x] == "." and not(text[x-1] == "S")  and (text[x+1] == " "or text[x+1].isupper())):
            sentences.append(current_sentence)
            current_sentence = ""
    return sentences


# # Running Code

# In[7]:


##Creating an array of all the sentences for each specific article
GS_sentences = get_sizeable_sentences(Get_Text("https://www.goldmansachs.com/insights/pages/as-bank-stress-grows-are-markets-signaling-a-us-recession.html"))
JPM_sentences = get_sizeable_sentences(Get_Text("https://www.jpmorgan.com/insights/research/market-outlook"))
BOA_sentences = get_sizeable_sentences(Get_Text("https://www.foxbusiness.com/economy/bank-america-still-forecasting-2023-recession-fed-action-not-enough-exec-warns"))
WF_sentences = get_sizeable_sentences(Get_Text("https://newsroom.wf.com/English/news-releases/news-release-details/2022/Wells-Fargo-Investment-Institute-2023-Outlook-A-Year-of-Recession-Recovery-and-Rebound/default.aspx"))
CITI_sentences = get_sizeable_sentences(Get_Text("https://www.reuters.com/article/global-economy-citigroup/citi-sees-lower-probability-of-global-recession-in-2023-idUKL4N3433F"))
MS_sentences = get_sizeable_sentences(Get_Text("https://www.cbsnews.com/news/morgan-stanley-stock-outlook-20-percent-drop/"))
USBC_sentences = get_sizeable_sentences(Get_Text("https://www.usbank.com/investing/financial-perspectives/market-news/economic-recovery-status.html#:~:text=The%20Federal%20Reserve%20continues%20to,by%20a%20strong%20job%20market."))
TD_sentences = get_sizeable_sentences(Get_Text("https://seekingalpha.com/article/4570388-not-in-recession-yet-but-growth-likely-to-slow"))
COB_sentences = get_sizeable_sentences(Get_Text("https://seekingalpha.com/article/4572920-capital-one-financial-shares-struggle-2023?mailingid=30374364&messageid=2800&serial=30374364.2229&utm_campaign=rta-stock-article&utm_medium=email&utm_source=seeking_alpha&utm_term=30374364.2229"))
SSC_sentences = get_sizeable_sentences(Get_Text("https://investors.statestreet.com/investor-news-events/press-releases/news-details/2022/State-Street-Global-Advisors-Survey-Inflation-Causing-Stress-and-Anxiety-Nearly-Half-of-Investors-Believe-It-Has-Not-Peaked-08-09-2022/default.aspx"))
HSBC_sentences = get_sizeable_sentences(Get_Text("https://www.gbm.hsbc.com/en-gb/feed/global-research/stalling-or-crawling"))
CHSW_sentences = get_sizeable_sentences(Get_Text("https://www.schwab.com/learn/story/outlook-overview#:~:text=In%20fact%2C%20the%20Fed%20is,increase%20from%20the%20current%20rate."))


# In[8]:


##Creates a sentiment analysis of each individual sentence and stores it in an array for each unique article
GS_sentiment = []
for x in GS_sentences:
    GS_sentiment += sentiment_analysis(x)

JPM_sentiment = []
for x in JPM_sentences:
    JPM_sentiment += sentiment_analysis(x)

BOA_sentiment = []
for x in BOA_sentences:
    BOA_sentiment += sentiment_analysis(x)

WF_sentiment = []
for x in WF_sentences:
    WF_sentiment += sentiment_analysis(x)
    
CITI_sentiment = []
for x in CITI_sentences:
    CITI_sentiment += sentiment_analysis(x)
    
MS_sentiment = []
for x in MS_sentences:
    MS_sentiment += sentiment_analysis(x)
    
USBC_sentiment = []
for x in USBC_sentences:
    USBC_sentiment += sentiment_analysis(x)
    
TD_sentiment = []
for x in TD_sentences:
    TD_sentiment += sentiment_analysis(x)
    
COB_sentiment = []
for x in COB_sentences:
    COB_sentiment += sentiment_analysis(x)
    
SSC_sentiment = []
for x in SSC_sentences:
    SSC_sentiment += sentiment_analysis(x)
    
HSBC_sentiment = []
for x in HSBC_sentences:
    HSBC_sentiment += sentiment_analysis(x)

CHSW_sentiment = []
for x in CHSW_sentences:
    CHSW_sentiment += sentiment_analysis(x)


# In[9]:


##Creates a dictionary for the sentiment analysis of each individual article based on individual sentences
GS_sentiment_data = aggragated_sentiments(GS_sentiment)
JPM_sentiment_data = aggragated_sentiments(JPM_sentiment)
BOA_sentiment_data = aggragated_sentiments(BOA_sentiment)
WF_sentiment_data = aggragated_sentiments(WF_sentiment)
CITI_sentiment_data = aggragated_sentiments(CITI_sentiment)
USBC_sentiment_data = aggragated_sentiments(USBC_sentiment)
TD_sentiment_data = aggragated_sentiments(TD_sentiment)
COB_sentiment_data = aggragated_sentiments(COB_sentiment)
SSC_sentiment_data = aggragated_sentiments(SSC_sentiment)
HSBC_sentiment_data = aggragated_sentiments(HSBC_sentiment)
CHSW_sentiment_data = aggragated_sentiments(CHSW_sentiment)


# In[10]:


#Creates the total array used to analyze overall market outlook
Total_Array = [GS_sentiment_data, JPM_sentiment_data, BOA_sentiment_data, WF_sentiment_data, 
               CITI_sentiment_data, USBC_sentiment_data, TD_sentiment_data, COB_sentiment_data, 
               SSC_sentiment_data, HSBC_sentiment_data, CHSW_sentiment_data]

#Prints out the total sentiment analyze using every individual article
total_sentiment(Total_Array)


# In[ ]:





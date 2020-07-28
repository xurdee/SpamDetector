"""
           A Spam detector script which also enables a user to detect and delete spam mails from their gmail inbox ( makes use
           of the awesome gmail api ;)

           version: 1.0.0
           authors : Samrat Dutta and Atul Aditya
           license : free to use
           contact : samratduttaofficial@gmail.com
                     atuladityasingh001@gmail.com
"""

from __future__ import print_function
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from google_auth_oauthlib.flow import InstalledAppFlow
from sklearn.model_selection import train_test_split
from google.auth.transport.requests import Request
from sklearn.naive_bayes import MultinomialNB
from googleapiclient.discovery import build
from nltk.tokenize import word_tokenize
from TextFormatter import TextFormatter
from collections import OrderedDict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import dateutil.parser as parser
from bs4 import BeautifulSoup
from apiclient import errors
import pandas as pd
import lxml.html
import datetime
import os.path
import base64
import pickle
import joblib
import lxml
import csv
import re
import os


def train_model(file_name):
    if file_name == '':
        file_name = 'spammails.csv'
    print(f'\nLooking for file {file_name}...')
    if os.path.exists(file_name):
        print(f'File {file_name} found !')
        cv = CountVectorizer()
        print(f'Reading from file now...')
        df = pd.read_csv(file_name)
        print('Done!')
        # Print the shape (Get the number of rows and cols)
        print(f'Number of rows and columns in the data frame are : ')
        print(df.shape)
        # Get the column names
        print('Names of columns in the data frame are: ')
        print(df.columns)
        # Checking for duplicates and removing them
        print(f'Dropping the duplicates...')
        df.drop_duplicates(inplace=True)
        print('Done!')
        X = df['text']
        y = df['spam']
        print('Text and Spam values of the dataframe are taken into X and Y respectively.')
        # Fit to data, then transform it.
        X = cv.fit_transform(X)
        print('X is fitted to data and transformed.')
        # Split data into 66% training & 33% testing data sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print('Data is split into 66% training & 33% testing data sets\n')

        # Naive Bayes Classifier
        # Create and train the Multinomial Naive Bayes classifier which is suitable for classification with discrete
        # features (e.g., word counts for text classification)
        # Also save the classifier model for future use (no need to train more than once

        print('Training of Model started...')
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        clf.score(x_test, y_test)
        print('Dumping trained model to a pkl file...')
        joblib.dump(clf, 'NB_spam_model.pkl')
        print('Dumped!')
        print('Model Trained successfully !')

        # Evaluate the model on the training data set
        print('Evaluating the model on the training data set...\n')

        pred = clf.predict(x_train)
        print(classification_report(y_train, pred))
        print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
        print()
        print('Accuracy: ', accuracy_score(y_train, pred))

        # Evaluate the model on the test data set
        print('Evaluating the model on the test data set...\n')
        pred = clf.predict(x_test)
        print(classification_report(y_test, pred))
        print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
        print()
        print('Accuracy: ', accuracy_score(y_test, pred))

        return clf
    else:
        show_err_msg(f'No such file : {file_name} found!')
        return 0


def refine_msg(message):

    # Removing the hyperlinks
    message = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', message)
    # Converting to a single line of strings
    message = " ".join(item.strip() for item in message.split('\n'))

    # Removing the punctuations and special characters
    message = re.sub(r'[^a-zA-Z0-9\d\s]', '', message)
    message = " ".join(item.strip() for item in message.split(' '))
    message_unique_words = set(message.split(' '))
    message = " ".join(list(message_unique_words))

    # Removing the stopwords and stemming the message
    words = word_tokenize(message)
    words = set(words)
    ps = PorterStemmer()
    message = " ".join([ps.stem(w) for w in words if w.lower() not in stopwords.words('english')])

    return message


def classify(message):
    try:
        cv_trained = open('CountVectorizer.pkl', 'rb')
        cv = joblib.load(cv_trained)
    except Exception as e:
        print(f'Error opening CV file : {e}')
        cv = CountVectorizer()
        df = pd.read_csv('spammails.csv')
        df.drop_duplicates(inplace=True)
        X = df['text']
        cv.fit_transform(X)
        joblib.dump(cv, 'CountVectorizer.pkl')

    try:
        nb_spam_model = open('nb_spam_model.pkl', 'rb')
        clf = joblib.load(nb_spam_model)
    except Exception as e:
        show_err_msg(f'Error opening pre trained Spam Model : {e}\n')
        print('Training a model now...\n')
        # When training from inside classify function 'spammails.csv' file will be used by default for training a model.
        # If this file is not found it will result in an error message and the control will be returned back to main.
        clf = train_model(file_name='')
        if clf == 0:
            show_err_msg(f'Error. The default spammails.csv file was not found. Training model failed!!.')
            return

    message = refine_msg(message)

    # check given data through classifier
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    return int(my_prediction[0])


SCOPES = ['https://mail.google.com/']  # for read-write
user_id = 'me'
listMails = []


def search(service, user_id, query):

    try:
        response = service.users().messages().list(userId=user_id, q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query, pageToken=page_token).execute()
            messages.extend(response['messages'])
        return messages
    except errors.HttpError as e:
        show_err_msg(f'An error occurred: {e}')


def is_spam(msg):
    return classify(msg)


# Cleans the message body...
def make_msg_body(mssg_body):

    # Removing the empty lines
    condensed_str = os.linesep.join([s for s in mssg_body.splitlines() if s.strip()])

    # Removing all the meta tags
    document = lxml.html.document_fromstring(condensed_str)
    condensed_str = document.text_content()

    # Removing duplicate lines
    condensed_str = '\n'.join(OrderedDict.fromkeys(condensed_str.split('\n')))

    return condensed_str


def get_mail_details(service, user_id, msg_id):

    temp_dict = {}
    message = service.users().messages().get(userId=user_id, id=msg_id, ).execute()  # Fetch the message using API
    payld = message['payload']  # Get payload of the Message
    headr = payld['headers']  # Get Header of the Payload

    for one in headr:  # Getting the Mail Subject
        if one['name'] == 'Subject':
            msg_subject = one['value']
            temp_dict['Subject'] = msg_subject
        else:
            pass

    for two in headr:  # Getting the Date on which mail was received
        if two['name'] == 'Date':
            msg_date = two['value']
            date_parse = (parser.parse(msg_date))
            m_date = (date_parse.date())
            temp_dict['Date'] = str(m_date)
        else:
            pass

    for three in headr:  # Getting the Sender of the Mail
        if three['name'] == 'From':
            msg_from = three['value']
            temp_dict['From'] = msg_from
        else:
            pass

    temp_dict['Snippet'] = message['snippet']  # Getting the Message Snippet

    # Fetching message body now...
    try:
        mssg_parts = payld['parts']  # fetching the message parts
        part_one = mssg_parts[0]  # fetching first element of the part
    except Exception as e:
        pass
    try:
        part_body = part_one['body']  # fetching body of the message
    except Exception as e:
        pass
    try:
        part_data = part_body['data']  # fetching data from the body
    except Exception as e:
        pass
    try:
        clean_one = part_data.replace("-", "+")  # decoding from Base64 to UTF-8
        clean_one = clean_one.replace("_", "/")  # decoding from Base64 to UTF-8
        clean_two = base64.b64decode(bytes(clean_one, 'UTF-8'))  # decoding from Base64 to UTF-8
        soup = BeautifulSoup(clean_two, "lxml")

        mssg_body = soup.body()
        mssg_body = str(mssg_body)  # Soup object needs to be type casted to String object

        clean_message = make_msg_body(mssg_body)
        temp_dict['Message_body'] = clean_message  # Setting the message body
    except Exception as e:
        # print(e)
        temp_dict['Message_body'] = message['snippet']

    print(temp_dict)

    return temp_dict


def check_gmail_inbox(query):

    global listMails

    if len(listMails) != 0:
        listMails.clear()
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    global service
    service = build('gmail', 'v1', credentials=creds)
    if query == 'NONE':  # make this if query == '' , to stop fetching without any query, otherwise this may have thousands of email and it will take way long time for google's server to finish
        pass
    else:
        messages = search(service=service, user_id=user_id, query=query)
        if messages:
            print(f'{len(messages)} emails found with query: {query}\n\n')
            file = open(r'report.txt', 'br+')
            file.seek(0)
            file.write((
                f'################################### Scan Report ({datetime.datetime.now()})################################\n\n').encode(
                'utf-8'))
            file.write(f'{len(messages)} mails found with your query: {query}\n\n'.encode('utf-8'))
            file.truncate()
            file.close()

            for mssg in messages:

                temp_dict = get_mail_details(service=service, user_id=user_id, msg_id=mssg['id'])
                temp_dict2 = {}

                if is_spam(temp_dict['Message_body']):
                    temp_dict[
                        'flag'] = 'SPAM'  # If the classifier returns 1 set the flag value as 'SPAM' otherwise 'HAM' later this can be used to differentiate ..
                else:
                    temp_dict['flag'] = 'HAM'

                temp_dict2[mssg['id']] = temp_dict
                # listMails has the details of all the mails found with the query entered by user
                listMails.append(
                    temp_dict2)  # This will create a list of nested dictionaries where each dictionary contains the details of a particular mail

            # Writing the results to report.txt
            file = open(r'report.txt', 'ab')
            for i in range(len(listMails)):
                mail = listMails[i]
                # mail_id = list(mail.keys())[0]
                # print(f'The mail id is : {mail_id}')
                mail_det = list(mail.values())[0]
                date = mail_det['Date']
                sender = mail_det['From']
                subject = mail_det['Subject']
                snippet = mail_det['Snippet']
                body = mail_det['Message_body']
                file.write(f'{i + 1})\n\n'.encode('utf-8'))
                file.write(f'Date received:   {date}.\n\n'.encode('utf-8'))
                file.write(f'From:   {sender}\n\n'.encode('utf-8'))
                file.write(f'Subject:   {subject}\n\n'.encode('utf-8'))
                file.write(f'Snippet:   {snippet}\n\n'.encode('utf-8'))
                file.write(f'Body:   \n{body}\n'.encode('utf-8'))
                if mail_det['flag'] == 'SPAM':
                    file.write(f'Classified as:   SPAM!\n\n'.encode('utf-8'))
                else:
                    file.write(f'Classified as:   HAM\n\n'.encode('utf-8'))

                file.write(
                    f'--------- x x x x x x x ---------- x x x x x x x ----------- x x x x x x x ------------- x x x x x x x -----------\n\n'.encode(
                        'utf-8'))

            file.close()
            show_mails(listMails)

        else:
            show_err_msg(f"There was no mail matching the query {query}")


def delete_mail_from_id(service, user_id, mail_id, list_mails):
    try:
        service.users().messages().delete(userId=user_id, id=mail_id).execute()
        print(f'\nMessage with id \'{mail_id}\' deleted!\n')

        # Reconstructing the 'listMails' list after the message is deleted...
        list_mails = [{k: v for k, v in d.items() if str(k) != mail_id} for d in list_mails]
        list_mails.remove({})

        return list_mails

    except Exception as e:
        show_err_msg(f'\nFailed to delete the mail! . Error {e}\n')
        return list_mails


def get_valid_delete_pos(list_mails):
    deleting_mail_pos = int(input('Enter position of the mail you want to delete (0 for none): '))
    if deleting_mail_pos <= len(list_mails) and deleting_mail_pos >= 0:
        deleting_mail_pos -= 1
    else:
        show_err_msg('Index error!. Please enter a valid position\n')
        deleting_mail_pos = get_valid_delete_pos(list_mails)
    return deleting_mail_pos


def show_mails(list_mails):
    for i in range(len(list_mails)):
        mail = list_mails[i]
        mail_det = list(mail.values())[0]
        mail_date = mail_det['Date']
        mail_sender = mail_det['From']
        mail_snipp = mail_det['Snippet']
        if mail_det['flag'] == 'SPAM':
            print(f'{i + 1}. Date: {mail_date} ,From: {mail_sender}  {mail_snipp.strip()}  <SPAM>')
        else:
            print(f'{i + 1}. Date: {mail_date} ,From: {mail_sender}  {mail_snipp.strip()}  <HAM>')

    deleting_mail_pos = get_valid_delete_pos(list_mails)
    if deleting_mail_pos != -1:
        list_mails = delete_mail_from_id(service=service, user_id=user_id,
                                         mail_id=list(list_mails[deleting_mail_pos].keys())[0], list_mails=list_mails)
        if len(list_mails) > 0:
            show_mails(list_mails)
        else:
            return
    else:
        return


def check_spam_from_user_input():  # delete this fn for app only for KV
    input_text = str(input('Enter the mail body :'))
    if input_text != '':
        input_text = refine_msg(input_text)
        if is_spam(msg=input_text):
            print("The given email is SPAM!! :(")
        else:
            print("The given email is HAM :)")
    else:
        return


def get_non_empty_filename():
    text = str(input('Enter the filename: '))
    if text == "":
        show_err_msg('Filename can\'t be empty!')
        text = get_non_empty_filename()
        return text
    else:
        return text


def retrieve_csv_filename():
    filename = get_non_empty_filename()
    li = filename.split('.')
    if len(li) > 1:
        if li[1] != 'csv':
            show_err_msg('Only csv file is accepted!')
            filename = retrieve_csv_filename()
            return filename
    else:
        filename = filename.strip().__add__('.csv')
    return filename


def check_spam_from_csv():  # delete this fn for app only for KV
    filename = retrieve_csv_filename()
    if os.path.exists(filename):
        with open(filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                if is_spam(refine_msg(row[0])):
                    print(f'\n{row[0]} : <SPAM>')
                else:
                    print(f'\n{row[0]} : <HAM>')
    else:
        filename = filename.split('.')[0]
        show_err_msg(f'No such file {filename} found!')
        return


def show_err_msg(msg):
    cprint = TextFormatter()
    cprint.cfg('r', '', 'b')
    cprint.out(msg+'\n')
    cprint.reset()


def show_welcome_message():
    cprint = TextFormatter()
    cprint.cfg('b', '', '')
    cprint.out('\n<=======================   ')
    cprint.cfg('y', '', 'b')
    cprint.out('SPAM EMAIL DETECTOR')
    cprint.cfg('b', '', 'i')
    cprint.out(' ---> A project by ')
    cprint.cfg('w', '', 'b')
    cprint.out('Samrat Dutta')
    cprint.cfg('b', '', 'i')
    cprint.out(' and ')
    cprint.cfg('w', '', 'b')
    cprint.out('Atul Aditya.')
    cprint.cfg('b', '', '')
    cprint.out('   ========================>\n')
    cprint.reset()


def show_menu():
    cprint = TextFormatter()
    cprint.cfg(fg='g', st='b')
    cprint.out('\n1. Train a machine learning model using a csv file.\n')
    cprint.out('2. Check if a mail body is spam or not manually.\n')
    cprint.out('3. Classify emails from a csv file.\n')
    cprint.out('4. Classify and delete mails from your Gmail inbox.\n')
    cprint.out('5. Exit.\n')
    cprint.reset()


if __name__ == '__main__':
    print(__doc__)

    show_welcome_message()
    while 1:
        show_menu()
        try:
            userChoice = int(input('\nEnter Your Choice: '))
            if userChoice == 1:
                file_name = retrieve_csv_filename()
                exit_code = train_model(file_name=file_name)
                if exit_code == 0:
                    continue
            elif userChoice == 2:
                check_spam_from_user_input()
            elif userChoice == 3:
                check_spam_from_csv()
            elif userChoice == 4:
                queryText = str(input(
                    'What do you want to search in your Gmail mailbox? (Leave empty to select all, type NONE to cancel operation: '))
                check_gmail_inbox(query=queryText)
            elif userChoice == 5:
                exit()
            else:
                show_err_msg('Invalid Choice!. Please enter a valid choice')
        except ValueError:
            show_err_msg('Only integer inputs are accepted!')
            continue


# This is how the list of mails looks like...
# listMails = [ { msg_id : {'Subject' : 'sample subject', 'Date' : '<27 July 2020>', 'From' : 'support@figma.com', 'Snippet' : 'sample message snippet...', 'Message_body' : 'samole message body...'} }, { msg_id : {} },{ msg_id : {} }, ......, { msg_id : {} } ]
# It's a list of nested dictionaries...


# This is How a gmail message looks like

# {
#   "id": "1555561f7b8e1sdf56b",
#   "threadId": "155552511dfsd83ce98",
#   "labelIds": [
#     "CHAT"
#   ],
#   "snippet": "This is a sample snippet...",
#   "historyId": "270812",
#   "internalDate": "1466016331704",
#   "payload": {
#     "partId": "",
#     "mimeType": "text/html",
#     "filename": "",
#     "headers": [
#       {
#         "name": "From",
#         "value": "\"Atul Aditya Singh\" <atuladityasingh001@gmail.com>"
#       }
#     ],
#     "body": {
#       "size": 2,
#       "data": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolor."
#     }
#   },
#   "sizeEstimate": 100
# }

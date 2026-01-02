
# import joblib
# from preprocessing import text_processing_pipeline

# pipe_nb = joblib.load(open("using_lstm/clf_model.pkl", "rb"))
# model = pipe_nb

# def predict_emotions(text):
#     return model.predict([text])

# def view_emotions(comments):
#     temp = []

#     for comm in comments:
#         temp.append(text_processing_pipeline(comm))

#     clean_comments = temp

#     labels = []

#     ang, fear, joy, neu, sad, love, sur = 0, 0, 0, 0, 0, 0, 0

#     for i in clean_comments:
#         emotion = predict_emotions(i)
#         labels.append(emotion)

#         if emotion == 'anger':
#             ang+=1
#         elif emotion == 'fear':
#             fear=+1
#         elif emotion == 'joy':
#             joy+=1
#         elif emotion == 'love':
#             love+=1
#         elif emotion == 'sadness':
#             sad+=1
#         elif emotion == 'surprise':
#             sur+=1
#         else:
#             neu+=1
#     e_no = [ang,love,fear,joy,sad,sur]
#     return len(clean_comments), ang, love, fear, joy, sad, sur, e_no, comments, labels


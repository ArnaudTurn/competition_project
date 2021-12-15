#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# API                                                                         #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-12                                                            #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################
import requests
from flask import Flask, jsonify, request, render_template, session, redirect
import pandas as pd
import joblib
import sys

app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route('/predict', methods=['GET'])
def predict():
    a = str(pd.DataFrame({"ok":[1,2,3]}))
    return render_template('simple.html',  tables=[a.to_html(classes='data')], titles=a.columns.values)

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    print("ok")
    app.run(debug=True, port = port)
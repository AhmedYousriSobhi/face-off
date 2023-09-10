"""
    Author: Ahmed Sobhi
    Department: Data Science
    Created_at: 2023-08-28
    Objective: Create a workspace directories for project.
"""

# Importing required libararies
import os

# Code Flow
if __name__ == '__main__':

    # Create required directories for data
    if not os.path.exists('data/'):
        os.makedirs('data/train')
        os.makedirs('data/test')
        os.makedirs('data/label')

    # Create reqiored directories for models
    if not os.path.exists('face_match'):
        os.makedirs('face_match/')
        
    # Create required directories for Documentation
    if not os.path.exists('docs/'):
        os.makedirs('docs/')

    # Create required directories for report
    if not os.path.exists('report/'):
        os.makedirs('report/plots')
        os.makedirs('report/reports')
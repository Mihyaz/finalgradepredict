import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import matplotlib.lines as mlines
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep = ";")
"""
instr: Instructor's identifier; values taken from {1,2,3}
class: Course code (descriptor); values taken from {1-13}
repeat: Number of times the student is taking this course; values taken from {0,1,2,3,...}
attendance: Code of the level of attendance; values from {0, 1, 2, 3, 4}
difficulty: Level of difficulty of the course as perceived by the student; values taken from {1,2,3,4,5}
Q1: The semester course content, teaching method and evaluation system were provided at the start.
Q2: The course aims and objectives were clearly stated at the beginning of the period.
Q3: The course was worth the amount of credit assigned to it.
Q4: The course was taught according to the syllabus announced on the first day of class.
Q5: The class discussions, homework assignments, applications and studies were satisfactory.
Q6: The textbook and other courses resources were sufficient and up to date.
Q7: The course allowed field work, applications, laboratory, discussion and other studies.
Q8: The quizzes, assignments, projects and exams contributed to helping the learning.
Q9: I greatly enjoyed the class and was eager to actively participate during the lectures.
Q10: My initial expectations about the course were met at the end of the period or year.
Q11: The course was relevant and beneficial to my professional development.
Q12: The course helped me look at life and the world with a new perspective.
Q13: The Instructor's knowledge was relevant and up to date.
Q14: The Instructor came prepared for classes.
Q15: The Instructor taught in accordance with the announced lesson plan.
Q16: The Instructor was committed to the course and was understandable.
Q17: The Instructor arrived on time for classes.
Q18: The Instructor has a smooth and easy to follow delivery/speech.
Q19: The Instructor made effective use of class hours.
Q20: The Instructor explained the course and was eager to be helpful to students.
Q21: The Instructor demonstrated a positive approach to students.
Q22: The Instructor was open and respectful of the views of students about the course.
Q23: The Instructor encouraged participation in the course.
Q24: The Instructor gave relevant homework assignments/projects, and helped/guided students.
Q25: The Instructor responded to questions about the course inside and outside of the course.
Q26: The Instructor's evaluation system (midterm and final questions, projects, assignments, etc.) effectively measured the course objectives.
Q27: The Instructor provided solutions to exams and discussed them with students.
Q28: The Instructor treated all students in a right and objective manner.

Q1-Q28 are all Likert-type, meaning that the values are taken from {1,2,3,4,5}
"""
"""data = data
[[
    "instr","class","nb.repeat","attendance","difficulty","Q1","Q2","Q3","Q4","Q5",
    "Q6","Q7","Q8","Q9","Q10","Q11","Q12","Q13","Q14","Q15","Q16","Q17","Q18","Q19",
    "Q20","Q21","Q22","Q23","Q24","Q25","Q26","Q27","Q28"
]]"""
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

first_grade = input("Enter your first grade ")
data["G1"][0] = first_grade

second_grade = input("Enter your second grade ")
data["G2"][0] = second_grade

study_time = input("Enter your study time ")
data["studytime"][0] = study_time

failures = input("Enter your failures ")
data["failures"][0] = failures

absences = input("Enter your absences ")
data["absences"][0] = absences

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""best = 0
for _ in range(400):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best:
       print(acc)
       best = acc
       with open("Q1.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("Q1.pickle", "rb")
linear = pickle.load(pickle_in)

#print('Co: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)
x_test[0][0] = first_grade
x_test[0][1] = second_grade
x_test[0][2] = study_time
x_test[0][3] = failures
x_test[0][4] = absences
predictions = linear.predict(x_test)

for x in range (len(predictions)):
   print("Predicted Final Grade: " + str(predictions[x]), x_test[x], "Final Grade: " +str( y_test[x]), "Error: " + str((predictions[x] - y_test[x])))

print("Your final grade can be ", predictions[0])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
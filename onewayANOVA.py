from scipy.stats import f_oneway

# Exam scores for each class
class_A = [85, 90, 88, 82, 87]
class_B = [76, 78, 80, 81, 75]
class_C = [92, 88, 94, 89, 90]

# Perform one-way ANOVA test
f_statistic, p_value = f_oneway(class_A, class_B, class_C)

# Print the results
print("F-statistic:", f_statistic)
print("p-value:", p_value)

# Interpret the results
alpha = 0.05  # significance level

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in the mean exam scores among the classes.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the mean exam scores among the classes.")

def factorial(n):
    reps = 1
    result = n
    while (reps < n):
        result = result * ((n- reps))
        reps = reps + 1
    return result

result = factorial(5)
print(result)


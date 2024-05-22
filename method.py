def separate_lines(text):
    return text.splitlines()

def SGE(Q, f):
    z_explore = "List all heuristic methods to solve this problem. Return them separated by new lines."
    z_decomp = "List all steps to use the heuristic method. Return them separated by new lines."
    z_feedback = "Give feedback to the proposed solution."
    z_integrate = "Integrate all previous findings and provide the final answer. Reuturn answer as list of customer indices to go."

    final_thoughts = []
    current_prompt = Q + ' ' + z_explore
    Q_N = f.predict(current_prompt)
    Q_N = separate_lines(Q_N)

    for n in range(len(Q_N)):
        Q_n = Q_N[n]
        current_prompt = Q + ' ' + Q_n + ' ' + z_decomp
        Q_n_K = f.predict(current_prompt)
        Q_n_K = separate_lines(Q_n_K)
        T_n_k_list = [""]

        for k in range(len(Q_n_K)):
            Q_n_k = Q_n_K[k]
            current_prompt = Q + ' ' + T_n_k_list[-1] + ' ' + Q_n_k
            T_n_k = f.predict(current_prompt)
            current_prompt = Q + ' ' + Q_n_k + ' ' + T_n_k + ' ' + z_feedback
            Q_n_k_feedback = f.predict(current_prompt)
            current_prompt = Q + ' ' + T_n_k + ' ' + Q_n_k_feedback
            T_n_k = f.predict(current_prompt)
            T_n_k_list.append(T_n_k)

        final_thoughts.append(T_n_k_list[-1])

    current_prompt = Q + ' ' + final_thoughts.join(" ") + z_integrate
    A = f.predict(current_prompt)

    return A
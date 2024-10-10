import numpy as np
import os
import matplotlib.pyplot as plt
import time

maxs = 254
dim = 15
file_num = 100
file_output_path = "output.txt"
output_folder = "output/"
temp_f_n = 11
unkn_f_n = 21


class McepData:
    def __init__(self):
        self.name = ""
        self.onso = ""
        self.flame = 0
        self.mcepdata = np.zeros((maxs, dim))


def read_mcep_file(filename):
    with open(filename, 'r') as file:
        mcepdata = McepData()
        mcepdata.name = file.readline().strip()
        mcepdata.onso = file.readline().strip()
        mcepdata.flame = int(file.readline().strip())
        for i in range(mcepdata.flame):
            mcepdata.mcepdata[i] = list(
                map(float, file.readline().strip().split()))
    return mcepdata


def calk_dis(template_file, unknown_file):
    d = np.zeros((maxs, maxs))
    g = np.zeros((maxs, maxs))

    for i in range(template_file.flame):
        for j in range(unknown_file.flame):
            d[i, j] = np.sum((template_file.mcepdata[i] -
                             unknown_file.mcepdata[j]) ** 2)
            d[i, j] = np.sqrt(d[i, j])

    g[0, 0] = d[0, 0]

    for i in range(1, template_file.flame):
        g[i, 0] = g[i - 1, 0] + d[i, 0]
    for j in range(1, unknown_file.flame):
        g[0, j] = g[0, j - 1] + d[0, j]

    for i in range(1, template_file.flame):
        for j in range(1, unknown_file.flame):
            a = g[i, j - 1] + d[i, j]
            b = g[i - 1, j - 1] + 2 * d[i, j]
            c = g[i - 1, j] + d[i, j]
            g[i, j] = min(a, b, c)

    path = [(template_file.flame - 1, unknown_file.flame - 1)]
    i, j = template_file.flame - 1, unknown_file.flame - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
            costs = [g[step] for step in steps]
            min_step = steps[np.argmin(costs)]
            i, j = min_step
        path.append((i, j))
    path.reverse()

    return g, g[template_file.flame - 1, unknown_file.flame - 1] / (template_file.flame + unknown_file.flame), path


def save_dtw_plot(g, template_idx, unknown_idx, path, template_flame, unknown_flame):
    plt.figure()
    plt.imshow(g[:template_flame, :unknown_flame], origin='lower', cmap='viridis',
               interpolation='none', extent=[0, unknown_flame, 0, template_flame])
    plt.colorbar(label='Cumulative Distance')
    plt.title("DPMatching Word Recognition \nTemplate Frame {} vs Unknown Frame {}".format(
        template_idx, unknown_idx))
    plt.xlabel('Template Frame Index')
    plt.ylabel('Unknown Frame Index')

    path = np.array(path)

    plt.plot(path[:, 1], path[:, 0], 'r')
    plt.axis('tight')

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "city{0:03d}_{1:03d}_vs_city{2:03d}_{3:03d}.png".format(
        temp_f_n, template_idx, unkn_f_n, unknown_idx)))
    print(os.path.join(output_folder, "city{0:03d}_{1:03d}_vs_city{2:03d}_{3:03d}.png".format(
        temp_f_n, template_idx, unkn_f_n, unknown_idx)))
    plt.close()


def main():
    count = 0
    first_comparison_done = False
    start = time.time()

    for h0 in range(file_num):
        temp_filename = "city_mcepdata/city{0:03d}/city{0:03d}_{1:03d}.txt".format(
            temp_f_n, h0 + 1)
        template_file = read_mcep_file(temp_filename)

        word_dis = np.zeros(file_num)

        for h in range(file_num):
            miti_filename = "city_mcepdata/city{0:03d}/city{0:03d}_{1:03d}.txt".format(
                unkn_f_n, h + 1)
            unknown_file = read_mcep_file(miti_filename)

            g, word_dis[h], path = calk_dis(template_file, unknown_file)

            if not first_comparison_done:
                save_dtw_plot(g, h0, h, path, template_file.flame,
                              unknown_file.flame)
                first_comparison_done = True

            elapsed_time = time.time() - start
            print("elapsed_time:{0:.3f}, h0:{1}, h:{2}, word_dis:{3:.3f}".format(
                elapsed_time, h0, h, word_dis[h]))

        word_dis_min = np.min(word_dis)
        num_match_fname = np.argmin(word_dis)

        if num_match_fname == h0:
            print("Matching")
            print("city{0:03d}_{1:03d}".format(temp_f_n, h0 + 1))
            print("city{0:03d}_{1:03d}".format(unkn_f_n, num_match_fname + 1))
            print("word distance : {}".format(word_dis_min))
            count += 1
            print(count)

        if num_match_fname != h0:
            print("NOT Matching")
            print("city{0:03d}_{1:03d}".format(temp_f_n, h0 + 1))
            print("city{0:03d}_{1:03d}".format(unkn_f_n, num_match_fname + 1))
            print("word distance : {}".format(word_dis_min))

    with open(file_output_path, 'a') as fp_output:
        fp_output.write("正答率{}%\n".format(count))
    print("\nファイルを作成しました。")
    print("正答率 {}% ".format(count))


if __name__ == "__main__":
    main()

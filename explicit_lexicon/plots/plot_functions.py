import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def boxplot_rewards(paths=('../results_fixed_lexicon/interval_15_alpha_5.0/',
                           '../results_dynamic_lexicon/interval_15_alpha_5.0/'),
                    runs=100):
    
    rewards_fixed = []
    rewards_dynamic = []

    for l, lexicon_type in enumerate(['fixed', 'dynamic']):
        path = paths[l]
        rewards = []
        for index in range(runs):
            rewards.append(np.load(path + 'rewards_' + str(index) + '.npy', allow_pickle=True)[:1500])

        if l == 0:
            rewards_fixed = np.mean(rewards, axis=0)
        elif l == 1:
            rewards_dynamic = np.mean(rewards, axis=0)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(4, 4))

    flierprops = dict(markeredgecolor='k')
    bpl = plt.boxplot([rewards_fixed, rewards_dynamic], positions=[0, 1.0], widths=0.6, flierprops=flierprops,
                      labels=['fixed', 'dynamic'])

    set_box_color(bpl, 'k')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('    ', fontsize=15)
    plt.title('learning performance', fontsize=15)
    plt.ylabel('reward', fontsize=15)
    plt.tight_layout()


def boxplot_ME_indices_bw(
        paths=('../results_fixed_lexicon/interval_15_alpha_5.0/', '../results_dynamic_lexicon/interval_15_alpha_5.0/'),
        runs=100,
        title=False):

    for l, lexicon_type in enumerate(['fixed', 'dynamic']):

        path = paths[l]
        rewards = []
        policies_lewis = []
        policies_reference = []
        for index in range(runs):
            rewards.append(np.load(path + 'rewards_' + str(index) + '.npy', allow_pickle=True)[:1500])
            policy_lewis = np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
            policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
            policies_lewis.append(policy_lewis[:1500])
            if lexicon_type == 'fixed':
                policies_reference.append(np.load(
                    path + 'policies_reference_' + str(index) + '.npy', allow_pickle=True)[-99:])
            elif lexicon_type == 'dynamic':
                policies_reference.append(np.load(
                    path + 'policies_ref_single_' + str(index) + '.npy', allow_pickle=True))
            index = index + 1

        if lexicon_type == 'fixed':
            possibly_correct_all = np.zeros((len(policies_lewis), len(policies_lewis[0])))
            for run in range(len(policies_lewis)):
                policy_run = policies_lewis[run]
                policy_run = np.squeeze(policy_run)
                possibly_correct = [(np.sum(policy_run[i, i:]) - (100 - i) / 100) / (i / 100)
                                    for i in range(len(policy_run))]
                possibly_correct_all[run, :] = np.array(possibly_correct)
            mean_lewis_fixed = np.mean(possibly_correct_all, axis=0)

        elif lexicon_type == 'dynamic':
            policy_length = len(policies_lewis[0])
            last_values = np.zeros((index, policy_length))
            for i in range(policy_length):
                for j in range(index):
                    last_values[j, i] = (policies_lewis[j][i][-1] - 1 / (i + 3)) / ((i + 3 - 1) / (i + 3))
            mean_lewis_dynamic = np.mean(last_values, axis=0)

        means_reference = [np.nanmean((policies_reference[i] - 0.5) / 0.5, axis=1)
                           for i in range(len(policies_reference))]
        mean_reference = np.array(means_reference)

        if lexicon_type == 'fixed':
            mean_ref_fixed = np.mean(mean_reference, axis=0)
        if lexicon_type == 'dynamic':
            mean_ref_dynamic = np.mean(mean_reference, axis=0)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(7, 4))

    flierprops = dict(markeredgecolor='k')
    bpl = plt.boxplot([mean_lewis_fixed, mean_ref_fixed], positions=[-0.3, 1.7], widths=0.6, flierprops=flierprops,
                      labels=['fixed', 'fixed'])
    bpr = plt.boxplot([mean_lewis_dynamic, mean_ref_dynamic], positions=[0.3, 2.3], widths=0.6, flierprops=flierprops,
                      labels=['dynamic', 'dynamic'])

    set_box_color(bpl, 'k')
    set_box_color(bpr, 'k')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('  ' + r'$\bf{general-context}$'
               + '                  ' + r'$\bf{specific-context}$', fontsize=15)
    plt.ylabel('$I_{ME}$', fontsize=15)
    if title:
        plt.title('ME bias', fontsize=15)
    plt.tight_layout()


def plot_multiple_iter(interval=(3, 6, 9, 12, 15), lexicon_type='fixed', mode='bias', addition=None, alpha=5.0,
                       runs=100):

    if lexicon_type == 'fixed':
        if addition:
            path = '../results_fixed_lexicon' + addition + '/interval_'
        else:
            path = '../results_fixed_lexicon/interval_'
        ylims = [0.94, 0.85]
    elif lexicon_type == 'dynamic':
        if addition:
            path = '../results_dynamic_lexicon' + addition + '/interval_'
        else:
            path = '../results_dynamic_lexicon/interval_'
        ylims = [0.89, 0.7]

    plt.figure(figsize=(18, 5.5))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    color = plt.cm.Reds(np.linspace(0.5, 1, len(interval)))

    for k, steps in enumerate(interval):

        full_path = path + str(steps) + '_alpha_' + str(alpha) + '/'

        rewards = []
        policies_lewis = []
        policies_reference = []

        for index in range(runs):
            rewards.append(np.load(full_path + 'rewards_' + str(index) + '.npy', allow_pickle=True)[:steps * 100])
            policy_lewis = np.load(full_path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
            policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
            policies_lewis.append(policy_lewis[:steps * 100])
            if lexicon_type == 'fixed':
                policies_reference.append(np.load(
                    full_path + 'policies_reference_' + str(index) + '.npy', allow_pickle=True)[-99:])
            elif lexicon_type == 'dynamic':
                policies_reference.append(np.load(
                    full_path + 'policies_ref_single_' + str(index) + '.npy', allow_pickle=True))
            index = index + 1

        mean_rewards = np.mean(rewards, axis=0)
        convstep = 19
        mean_convolved = np.convolve(mean_rewards, (1 / convstep) * np.ones(convstep), mode='valid')
        ax1.plot(np.arange(convstep // 2, len(mean_rewards) - convstep // 2), mean_convolved, color=color[k])

        if lexicon_type == 'fixed':
            possibly_correct_all = np.zeros((len(policies_lewis), len(policies_lewis[0])))
            for run in range(len(policies_lewis)):
                policy_run = policies_lewis[run]
                policy_run = np.squeeze(policy_run)
                if mode == 'bias':
                    possibly_correct = [np.sum(policy_run[i, i:]) for i in range(len(policy_run))]
                elif mode == 'index':
                    possibly_correct = [(np.sum(policy_run[i, i:]) - (100 - i) / 100) / (i / 100) for i in
                                        range(len(policy_run))]
                possibly_correct_all[run, :] = np.array(possibly_correct)
            mean = np.mean(possibly_correct_all, axis=0)
            std = np.std(possibly_correct_all, axis=0)
            ax2.errorbar(range(1, len(mean) + 1), mean, yerr=std, errorevery=5, color=color[k])

        elif lexicon_type == 'dynamic':
            policy_length = len(policies_lewis[0])
            last_values = np.zeros((index, policy_length))
            for i in range(policy_length):
                for j in range(index):
                    if mode == 'bias':
                        last_values[j, i] = policies_lewis[j][i][-1]
                    elif mode == 'index':
                        last_values[j, i] = (policies_lewis[j][i][-1] - 1 / (i + 3)) / ((i + 3 - 1) / (i + 3))

            mean = np.mean(last_values, axis=0)
            std = np.std(last_values, axis=0)
            ax2.errorbar(range(3, len(mean) + 3), mean, yerr=std, errorevery=5, color=color[k])

        if mode == 'bias':
            means_reference = [np.nanmean(policies_reference[i], axis=1) for i in range(len(policies_reference))]
            ylim = ylims[0]
        elif mode == 'index':
            means_reference = [np.nanmean((policies_reference[i] - 0.5) / 0.5, axis=1) for i in
                               range(len(policies_reference))]
            ylim = ylims[1]

        mean_reference = np.array(means_reference)
        mean_across_runs = np.mean(mean_reference, axis=0)
        std_across_runs = np.std(mean_reference, axis=0)
        ax3.errorbar(np.arange(3, len(mean_across_runs) + 3), mean_across_runs,
                     yerr=std_across_runs, errorevery=5, color=color[k])

    ax3.set_ylim(bottom=ylim)

    ax1.set_xlabel('epoch', fontsize=15)
    ax1.set_ylabel('reward', fontsize=15)
    ax1.set_title('learning performance', fontsize=15)

    ax2.set_xlabel('novel word', fontsize=15)
    ax2.set_title('ME bias general-context ', fontsize=15)
    if mode == 'bias':
        ax2.set_ylabel('correct selection probability', fontsize=15)
    elif mode == 'index':
        ax2.set_ylabel('$I_{ME}$', fontsize=15)

    ax3.set_xlabel('novel word', fontsize=15)
    ax3.legend(['interval ' + str(i) for i in [3, 6, 9, 12, 15]], fontsize=15)
    ax3.set_title('ME bias specific-context', fontsize=15)
    if mode == 'bias':
        ax3.set_ylabel('correct selection probability', fontsize=15)
    elif mode == 'index':
        ax3.set_ylabel('$I_{ME}$', fontsize=15)
    plt.legend(['interval ' + str(i) for i in [3, 6, 9, 12, 15]], fontsize=15,
               loc='upper center', bbox_to_anchor=(-0.72, -0.15), fancybox=True,
               shadow=True, ncol=5)

    if lexicon_type == 'fixed':
        plt.suptitle(r'     $\bf{fixed\: lexicon}$' + '\n', fontsize=18)
    elif lexicon_type == 'dynamic':
        plt.suptitle(r'     $\bf{dynamic\: lexicon}$' + '\n', fontsize=18)
    plt.show()


def plot_learning_success(mode, time_spans=(10, 20, 30), addition='', alpha=5.0, runs=500,
                          threshold=0.99, ME_threshold=0.5, smallLR=True, min_sample_size=5):
    if mode == 'fixed':
        if smallLR:
            path = '../results_fixed_lexicon' + addition + '/interval_1_alpha_' + str(alpha) + '_small_learning_rate/'
        else:
            path = '../results_fixed_lexicon' + addition + '/interval_1_alpha_' + str(alpha) + '/'
    elif mode == 'dynamic':
        path = '../results_dynamic_lexicon' + addition + '/interval_1_alpha_' + str(alpha) + '/'

    policies_lewis = []
    counts = []

    for index in range(runs):
        policy_lewis = np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
        policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
        policies_lewis.append(policy_lewis)
        counts.append(np.load(path + 'counts_' + str(index) + '.npy', allow_pickle=True)[0:1100])

    for span in time_spans:
        plot_learning_success_time_span(policies_lewis, counts, runs=runs, min_sample_size=min_sample_size,
                                        maxdur=span, interval=1, ME_threshold=ME_threshold, mode=mode,
                                        learning_threshold=threshold)


def plot_learning_success_time_span(policies, counts, runs,
                                    n=100,
                                    maxdur=None,
                                    interval=1,
                                    learning_threshold=0.9,
                                    ME_threshold=0.,
                                    min_sample_size=1,
                                    mode='fixed'):
    if mode == 'fixed':
        iterations, range_min, plot_range_min, range_max = n, 0, 1, n
    elif mode == 'dynamic':
        iterations, range_min, plot_range_min, range_max = n - 2, 3, 3, n

    # iterate over runs and save ME indices as well as learning durations (with Inf if not learned)
    all_ME_indices = np.zeros((iterations, runs))
    all_learning_durations = np.zeros((iterations, runs))

    for run in range(runs):

        count = counts[run]
        policy = np.squeeze(policies[run])
        if mode == 'fixed':
            ME_index = np.array([(np.sum(policy[i, i:]) - (n - i) / n) / (i / n) for i in range(range_min, range_max)])
        elif mode == 'dynamic':
            ME_index = np.array(
                [(np.sum(policy[i - 3][-1]) - 1 / i) / ((i - 1) / i) for i in range(range_min, range_max + 1)])

        all_ME_indices[:, run] = ME_index

        learning_durations = np.empty((iterations))
        learning_durations[:] = np.Inf

        for referent in range(range_min, range_max):
            if maxdur is None:
                end = len(count)
            else:
                end = referent * interval + maxdur
            for epoch in range(referent * interval, end):
                sum_over_epochs = np.sum(count[epoch:, referent], axis=0)
                correct_classification = np.sum(sum_over_epochs[[0, 2]]) / np.sum(sum_over_epochs)
                if correct_classification > learning_threshold:
                    learning_durations[referent - range_min] = epoch - referent * interval + 1
                    break
        all_learning_durations[:, run] = learning_durations

    # extract the percentage of learned items for the two conditions as well as the learning times for those
    # learned samples --> store the results referent wise
    mean_duration_bias = []
    mean_duration_no_bias = []
    learned_bias = []
    learned_no_bias = []

    for referent in range(iterations):
        bias = all_ME_indices[referent, :]
        duration_bias = all_learning_durations[referent, bias > ME_threshold]
        duration_no_bias = all_learning_durations[referent, bias <= ME_threshold]
        duration_bias_learned = duration_bias[duration_bias < np.Inf]
        duration_no_bias_learned = duration_no_bias[duration_no_bias < np.Inf]
        if len(duration_bias) < min_sample_size:
            learned_bias.append(np.NaN)
            mean_duration_bias.append(np.NaN)
        else:
            learned_bias.append(len(duration_bias_learned) / len(duration_bias))
            mean_duration_bias.append(np.nanmean(duration_bias_learned))
        if len(duration_no_bias) < min_sample_size:
            learned_no_bias.append(np.NaN)
            mean_duration_no_bias.append(np.NaN)
        else:
            learned_no_bias.append(len(duration_no_bias_learned) / len(duration_no_bias))
            mean_duration_no_bias.append(np.nanmean(duration_no_bias_learned))

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(plot_range_min, plot_range_min + iterations), learned_bias, marker='v', alpha=0.8, color='b')
    plt.scatter(np.arange(plot_range_min, plot_range_min + iterations), learned_no_bias, marker='^', alpha=0.8,
                color='r')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('sample', fontsize=15)
    plt.ylabel('proportion learned', fontsize=15)
    plt.legend(['$I_{ME} > ' + str(ME_threshold) + '$', '$I_{ME} < ' + str(ME_threshold) + '$'], loc='lower left',
               fontsize=15)
    plt.title('learning success', fontsize=18)

    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(plot_range_min, plot_range_min + iterations), mean_duration_bias, marker='v', alpha=0.8,
                color='b')
    plt.scatter(np.arange(plot_range_min, plot_range_min + iterations), mean_duration_no_bias, marker='^', alpha=0.8,
                color='r')
    plt.legend(['$I_{ME} > ' + str(ME_threshold) + '$', '$I_{ME} < ' + str(ME_threshold) + '$'], loc='upper left',
               fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('sample', fontsize=15)
    plt.ylabel('duration (epochs)', fontsize=15)
    plt.title('learning duration', fontsize=18)
    if maxdur is None:
        plt.suptitle('maximum learning duration', fontsize=18)
    else:
        plt.suptitle('       ' + r'$\bf{time\:span: }$' + r"$\bf{" + str(maxdur) + "}$" + r' $\bf{epochs}$',
                     fontsize=18)
    fig.tight_layout(pad=2)
    plt.show()


def plot_general_context_policies(paths=('../results_fixed_lexicon/interval_15_alpha_5.0/',
                                         '../results_fixed_lexicon/interval_1_alpha_5.0/',
                                         '../results_dynamic_lexicon/interval_15_alpha_5.0/',
                                         '../results_dynamic_lexicon/interval_1_alpha_5.0/'),
                                  runs=100):

    mode = ['fixed', 'fixed', 'dynamic', 'dynamic']
    all_lewis = []
    all_reference = []
    for i, path in enumerate(paths):
        policies_lewis = []
        policies_reference = []
        for index in range(runs):
            if mode[i] == 'fixed':
                policies_lewis.append(np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True))
                policies_reference.append(
                    np.load(path + 'policies_reference_' + str(index) + '.npy', allow_pickle=True)[-99:]
                )
            elif mode[i] == 'dynamic':
                policy_lewis = np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
                policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
                policies_lewis.append(policy_lewis)
                policies_reference.append(
                    np.load(path + 'policies_ref_single_' + str(index) + '.npy', allow_pickle=True)
                )
        all_lewis.append(policies_lewis)
        all_reference.append(policies_reference)

    fig = plt.figure(figsize=(12, 10))
    for ax, policies in enumerate(all_lewis[0:2]):

        if ax == 0:
            gca = plt.gca()
            gca.text(-0.1, 1.15, 'A', transform=gca.transAxes, fontsize=25, fontweight='bold', va='top', ha='right')
        plt.subplot(2, 2, ax + 1)
        policies_mean = np.mean(np.array(policies), axis=0)
        policies_mean = np.squeeze(policies_mean)
        x = np.arange(0, len(policies_mean))
        im = plt.imshow(policies_mean, cmap='jet', vmin=0., vmax=0.05)
        plt.plot(x, x, linewidth=3, color='k', linestyle='dashed')
        plt.xticks(ticks=[0, 19, 39, 59, 79, 99], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.yticks(ticks=[0, 19, 39, 59, 79, 99], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.xlabel('object', fontsize=20)
        if ax == 0:
            plt.ylabel(r'$\bf{fixed\: lexicon}$' + '\n \n novel word', fontsize=20)
            plt.title(r'$\bf{optimal}$' + '\n', fontsize=20)
        else:
            plt.ylabel('novel word', fontsize=20)
            plt.title(r'$\bf{suboptimal}$' + '\n', fontsize=20)

    for ax, policies in enumerate(all_lewis[2:4]):
        plt.subplot(2, 2, ax + 3)
        mean = np.mean(policies, axis=0)
        policies_array = np.empty((98, 100))
        policies_array[:] = np.nan
        for i in range(len(mean)):
            len_tmp = len(mean[i])
            policies_array[i, 0:len_tmp] = mean[i]
        im = plt.imshow(policies_array, cmap='jet', vmax=0.05)
        plt.xticks(ticks=[0, 19, 39, 59, 79, 99], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.yticks(ticks=[-2, 17, 37, 57, 77, 97], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.xlabel('object', fontsize=20)
        if ax == 0:
            plt.ylabel(r'$\bf{dynamic\: lexicon}$' + '\n \n novel word', fontsize=20)
        else:
            plt.ylabel('novel word', fontsize=20)

    plt.text(-165, -160, 'A', fontsize=35, fontweight='bold', va='top', ha='right')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.2, 0.02, 0.5])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=15)
    cb.set_label('P (object | novel word)', fontsize=20, labelpad=15)
    cb.ax.set_yticklabels([0.01, 0.02, 0.03, 0.04, '0.05\n (cap)'])
    plt.suptitle(r'$\bf{general-context}$', fontsize=22, x=0.46, y=1.01)
    plt.show()


def plot_specific_context_policies(paths=('../results_fixed_lexicon/interval_15_alpha_5.0/',
                                          '../results_fixed_lexicon/interval_1_alpha_5.0/',
                                          '../results_dynamic_lexicon/interval_15_alpha_5.0/',
                                          '../results_dynamic_lexicon/interval_1_alpha_5.0/'),
                                   runs=100):

    mode = ['fixed', 'fixed', 'dynamic', 'dynamic']
    all_lewis = []
    all_reference = []
    for i, path in enumerate(paths):
        policies_lewis = []
        policies_reference = []
        for index in range(runs):
            if mode[i] == 'fixed':
                policies_lewis.append(
                    np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
                )
                policies_reference.append(
                    np.load(path + 'policies_reference_' + str(index) + '.npy', allow_pickle=True)[-99:]
                )
            elif mode[i] == 'dynamic':
                policy_lewis = np.load(path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
                policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
                policies_lewis.append(policy_lewis)
                policies_reference.append(
                    np.load(path + 'policies_ref_single_' + str(index) + '.npy', allow_pickle=True)
                )
        all_lewis.append(policies_lewis)
        all_reference.append(policies_reference)

    fig = plt.figure(figsize=(12, 10))
    for ax, policies in enumerate(all_reference):
        plt.subplot(2, 2, ax + 1)
        policies_mean = np.mean(np.array(policies), axis=0)
        x = np.arange(len(policies_mean))
        im = plt.imshow(policies_mean, cmap='jet', vmin=0.5, vmax=1.)
        plt.xticks(ticks=[0, 19, 39, 59, 79, 99], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.yticks(ticks=[-2, 17, 37, 57, 77, 97], labels=[1, 20, 40, 60, 80, 100], fontsize=15)
        plt.xlabel('distractor', fontsize=20)
        if ax == 0:
            plt.ylabel(r'$\bf{fixed\: lexicon}$' + ' \n \n target', fontsize=20)
            plt.title(r'$\bf{optimal}$' + '\n', fontsize=20)
        if ax == 1:
            plt.title(r'$\bf{suboptimal}$' + '\n', fontsize=20)
        if ax == 2:
            plt.ylabel(r'$\bf{dynamic\: lexicon}$' + '\n \n target', fontsize=20)
        elif ax == 1 or ax == 3:
            plt.ylabel('target', fontsize=20)

    plt.text(-165, -160, 'B', fontsize=35, fontweight='bold', va='top', ha='right')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.2, 0.02, 0.5])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label('P (target | novel word)', fontsize=20, labelpad=-20)
    cb.ax.set_yticklabels(['0.5\n(chance)', 0.6, 0.7, 0.8, 0.9, 1.0])
    cb.ax.tick_params(labelsize=15)
    plt.suptitle(r'$\bf{specific-context}$', fontsize=22, x=0.46, y=1.01)
    plt.show()


def plot_multiple_iter_k15(mode='index', path_addition='', alpha=5.0, runs=100, ablation=False, lims=None):

    if ablation:
        lexica = ['fixed', 'fixed']
        if not lims:
            lims = [0.36, 1.04]
            ticks = [0.4, 0.6, 0.8, 1.0]
        else: 
            ticks = np.round(np.arange(lims[0], lims[1], 0.2), 2)
        formatter = 1
    else:
        lexica = ['fixed', 'dynamic']
        if not lims: 
            lims = [0.74, 1.01]
            ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        else: 
            ticks = np.round(np.arange(lims[0], lims[1], 0.2), 2)
        formatter = 2

    for l, lexicon_type in enumerate(lexica):

        plt.figure(figsize=(11, 6))
        
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)

        if not ablation:
            full_path = ('../results_' + lexicon_type + '_lexicon' + path_addition +
                         '/interval_15' + '_alpha_' + str(alpha) + '/')
        else:
            full_path = ['../results_fixed_lexicon/pragmatic-learning_literal-inference/interval_15_alpha_5.0/',
                         '../results_fixed_lexicon/literal-learning_pragmatic-inference/interval_15_alpha_5.0/'][l]
        rewards = []
        policies_lewis = []
        policies_reference = []
        for index in range(runs):
            rewards.append(np.load(full_path + 'rewards_' + str(index) + '.npy', allow_pickle=True)[:1500])
            policy_lewis = np.load(full_path + 'policies_lewis_' + str(index) + '.npy', allow_pickle=True)
            policy_lewis = np.array([policy_lewis[i][0] for i in range(len(policy_lewis))])
            policies_lewis.append(policy_lewis[:1500])
            if lexicon_type == 'fixed':
                policies_reference.append(np.load(
                    full_path + 'policies_reference_' + str(index) + '.npy', allow_pickle=True)[-99:])
            elif lexicon_type == 'dynamic':
                policies_reference.append(np.load(
                    full_path + 'policies_ref_single_' + str(index) + '.npy', allow_pickle=True))
            index = index + 1

        mean_rewards = np.mean(rewards, axis=0)
        ax1.plot(mean_rewards, color='k', linewidth=0.8)
        
        ax1.set_ylim([0.84, 1.01])
        ax1.set_yticks([0.85, 0.90, 0.95, 1.00])
        ax1.set_yticklabels(ticks, fontsize=12)
        ax1.set_xticks([1, 500, 1000, 1500])
        ax1.set_xticklabels([1, 500, 1000, 1500], fontsize=12)
        ax1.set_xlabel('epoch', fontsize=14)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.' + str(2) + 'f'))
        ax1.set_ylabel('$R$', fontsize=15, labelpad=-3, color='k')
        ax1.set_title('reward', fontsize=14)

        if lexicon_type == 'fixed':
            possibly_correct_all = np.zeros((len(policies_lewis), len(policies_lewis[0])))
            for run in range(len(policies_lewis)):
                policy_run = policies_lewis[run]
                policy_run = np.squeeze(policy_run)
                if mode == 'bias':
                    possibly_correct = [np.sum(policy_run[i, i:]) for i in range(len(policy_run))]
                elif mode == 'index':
                    possibly_correct = [(np.sum(policy_run[i, i:]) - (100 - i) / 100) / (i / 100) for i in
                                        range(len(policy_run))]
                possibly_correct_all[run, :] = np.array(possibly_correct)
            mean = np.mean(possibly_correct_all, axis=0)
            std = np.std(possibly_correct_all, axis=0)
            eb1 = ax2.errorbar(range(1, len(mean)*15 + 1, 15), mean, yerr=std, linestyle='dotted', 
                              color='k', errorevery=15)
            eb1[-1][0].set_linestyle('dotted')

        elif lexicon_type == 'dynamic':
            policy_length = len(policies_lewis[0])
            last_values = np.zeros((index, policy_length))
            for i in range(policy_length):
                for j in range(index):
                    if mode == 'bias':
                        last_values[j, i] = policies_lewis[j][i][-1]
                    elif mode == 'index':
                        last_values[j, i] = (policies_lewis[j][i][-1] - 1 / (i + 3)) / ((i + 3 - 1) / (i + 3))

            mean = np.mean(last_values, axis=0)
            std = np.std(last_values, axis=0)
            eb1 = ax2.errorbar(range(3, len(mean) * 15 + 3, 15), mean, yerr=std, errorevery=15, color='k',
                               linestyle='dotted')
            eb1[-1][0].set_linestyle('dotted')
            
        if mode == 'bias':
            means_reference = [np.nanmean(policies_reference[i], axis=1) for i in range(len(policies_reference))]
        elif mode == 'index':
            means_reference = [np.nanmean((policies_reference[i] - 0.5) / 0.5, axis=1) for i in
                               range(len(policies_reference))]

        mean_reference = np.array(means_reference)
        mean_across_runs = np.mean(mean_reference, axis=0)
        std_across_runs = np.std(mean_reference, axis=0)
        eb2 = ax2.errorbar(np.arange(3, len(mean_across_runs)*15 + 3, 15), mean_across_runs, color='k',
                           yerr=std_across_runs, errorevery=14, linestyle='dashed')
        eb2[-1][0].set_linestyle('dashed')

        ax2.set_ylim(lims)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(ticks, fontsize=12)
        ax2.set_xticks([1, 500, 1000, 1500])
        ax2.set_xticklabels([1, 500, 1000, 1500], fontsize=12)
        ax2.tick_params(axis ='y')
        ax2.set_ylabel('$I_{ME}$', fontsize=15, labelpad=+5, color='k')
        ax2.set_xlabel('epoch', fontsize=14)
        ax2.set_title('ME bias', fontsize=14)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.' + str(formatter) + 'f'))
        ax2.legend(['general-context', 'specific-context'], fontsize=13)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3)
        
        if ablation:
            plt.suptitle(['pragmatic learning - literal inference', 'literal learning - pragmatic inference'][l], 
                         fontsize=15)
        else:
            plt.suptitle(['fixed lexicon \n', 'dynamic lexicon \n'][l], fontsize=15)

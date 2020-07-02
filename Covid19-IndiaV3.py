import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DateTime as dt
import os


# Function definitions
def doublingRate(di, vector, title, state, figno, day, output_path):
    cdr_interval = di  # Case or Death doubling interval
    cum_cases = vector.to_numpy()  # Cumulative cases converted to numpy
    cdr_pts = []  # Doubling rate points
    cdr_idx = np.argmax(cum_cases > 20) + cdr_interval  # n+7 idx after 20 case

    if cum_cases[-1] > 20:
        for cdr in range(cdr_idx, len(cum_cases)):
            cdr_pts.append(cdr_interval / (np.log2(cum_cases[cdr]) - np.log2(cum_cases[cdr - cdr_interval])))
        plt.figure(figno)
        plt.title(title + ' ' + state)
        plt.plot(day[cdr_idx:], cdr_pts[:], 'o-')
        plt.xlabel('Date')
        plt.ylabel(title)
        plt.xticks(day[cdr_idx::2], rotation=90)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.savefig(output_path + '\\' + title)


# State Selector
state = 'West Bengal'


def Covid19(stated):
    state = stated
    # Update this on your computer to have the correct output path
    output_path = 'C:\\Users\\masoo\\PycharmProjects\\Covid-19\\Results_India\\' + state
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    # Set this path on your computer
    read_file_cases = r'C:\Users\masoo\PycharmProjects\Covid-19\covid_19_india.csv'
    # Set this path on your computer
    read_file_test = r'C:\Users\masoo\PycharmProjects\Covid-19\StatewiseTestingDetails.csv'
    df_cases = pd.read_csv(read_file_cases)
    df_test = pd.read_csv(read_file_test)

    # Filter data-frame for state
    dfc = df_cases[df_cases["State/UnionTerritory"] == state]
    dft = df_test[df_test["State"] == state]

    # Day
    day = []
    for dates in dfc.Date:
        dayT = dates[0:5]
        day.append(dayT)

    # 1 Daily new cases
    cum_cases = dfc.Confirmed
    new_cases = [j - i for i, j in zip(cum_cases[:-1], cum_cases[1:])]
    new_cases.insert(0, cum_cases.values[0])  # Subtraction eats away first value, copy from cumulative list
    new_cases = pd.Series(new_cases)  # Converting list to Series
    dfc.insert(len(dfc.columns), "new_cases", new_cases)
    # Alternate - Slightly less elegant
    # new_cases2 = np.diff(cum_cases)
    case_1 = np.argmax(new_cases > 0)
    new_cases_roll = new_cases.rolling(5).mean()
    y_pos = range(len(day))
    plt.figure(1)
    plt.title('Daily new cases' + ' - ' + state)
    plt.bar(day[case_1:], new_cases[case_1:])
    # Rotation of the bars names
    plt.plot(day[case_1:], new_cases_roll[case_1:], 'r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Number of cases / day')
    plt.xticks(day[case_1::2], rotation=90)
    leg1 = '5 day moving average - ' + state
    leg2 = 'Number of cases / day - ' + state
    plt.legend([leg1, leg2])
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'NewCases.png')

    # 2 Cumulative cases
    plt.figure(2)
    plt.suptitle('Cumulative Cases' + ' - ' + state)
    plt.subplot(2, 1, 1)
    plt.grid(True, which="both")
    plt.semilogy(day[case_1:], cum_cases[case_1:])
    plt.ylim([100, max(cum_cases) + 1000])
    plt.xlabel('Date')
    plt.ylabel('Cumulative cases (Semilog plot)')
    plt.xticks(day[case_1::2], rotation=90)
    plt.subplot(2, 1, 2)
    plt.grid(True, which="both")
    plt.plot(day[case_1:], cum_cases[case_1:])
    plt.ylim([100, max(cum_cases) + 1000])
    plt.xlabel('Date')
    plt.ylabel('Cumulative cases')
    plt.xticks(day[case_1::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'CumulativeCases.png')

    # 3 Deaths (New Deaths)
    cum_deaths = dfc['Deaths']
    new_deaths = [j - i for i, j in zip(cum_deaths[:-1], cum_deaths[1:])]
    new_deaths.insert(0, cum_deaths.values[0])
    dfc.insert(len(dfc.columns), "new_deaths", new_deaths)
    new_deaths_roll = dfc.new_deaths.rolling(5).mean()
    plt.figure(3)
    plt.grid(True, which="both")
    plt.title('New deaths' + ' - ' + state)
    plt.bar(day[case_1:], new_deaths[case_1:])
    plt.plot(day[case_1:], new_deaths_roll[case_1:], 'r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Number of deaths / day')
    leg1 = '5 day moving average - '
    leg2 = 'Number of deaths / day - '
    plt.legend([leg1, leg2])
    plt.xticks(day[case_1::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'NewDeaths.png')

    # 4 Death cumulative
    plt.figure(4)
    plt.suptitle('Cumulative Deaths' + ' - ' + state)
    plt.subplot(2, 1, 1)
    plt.grid(True, which="both")
    plt.semilogy(day[case_1:], cum_deaths[case_1:])
    plt.ylim([0.1, max(cum_deaths) + (0.01 * max(cum_deaths))])
    plt.xlabel('Date')
    plt.ylabel('Cumulative deaths (Semilog plot)')
    plt.xticks(day[case_1::2], rotation=90)
    plt.subplot(2, 1, 2)
    plt.grid(True, which="both")
    plt.plot(day[case_1:], cum_deaths[case_1:])
    plt.ylim([0.1, max(cum_deaths) + (0.01 * max(cum_deaths))])
    plt.xlabel('Date')
    plt.ylabel('Cumulative deaths')
    plt.xticks(day[case_1::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'CumulativeDeaths.png')

    # 5 Recoveries
    cum_recov = dfc.Cured
    new_recov = [j - i for i, j in zip(cum_recov[:-1], cum_recov[1:])]
    new_recov.insert(0, cum_recov.values[0])  # Recover the first reported recovery
    dfc.insert(len(dfc.columns), "new_recoveries", new_recov)
    new_recov = pd.Series(new_recov)
    recov_1 = np.argmax(new_recov > 0)
    new_recov_roll = dfc.new_recoveries.rolling(5).mean()
    y_pos = range(len(day))
    plt.figure(5)
    plt.title('Daily new Recoveries' + ' - ' + state)
    plt.bar(day[recov_1:], new_recov[recov_1:])
    # Rotation of the bars names
    plt.plot(day[recov_1:], new_recov_roll[recov_1:], 'r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Number of recoveries / day')
    plt.xticks(day[recov_1::2], rotation=90)
    leg1 = '5 day moving average - ' + state
    leg2 = 'Number of recoveries / day - ' + state
    plt.legend([leg1, leg2])
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'DailyRecovs.png')

    # 6 Cumulative Recoveries
    plt.figure(6)
    plt.suptitle('Cumulative Recoveries' + ' - ' + state)
    plt.subplot(2, 1, 1)
    plt.grid(True, which="both")
    plt.semilogy(day[recov_1:], cum_recov[recov_1:])
    plt.ylim([0.1, max(cum_recov) + (0.01 * max(cum_recov))])
    plt.xlabel('Date')
    plt.ylabel('Cumulative recoveries (Semilog plot)')
    plt.xticks(day[recov_1::2], rotation=90)
    plt.subplot(2, 1, 2)
    plt.grid(True, which="both")
    plt.plot(day[recov_1:], cum_recov[recov_1:])
    plt.ylim([0.1, max(cum_recov) + (0.01 * max(cum_recov))])
    plt.xlabel('Date')
    plt.ylabel('Cumulative recoveries')
    plt.xticks(day[recov_1::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'CumRecovs.png')

    # 7 Cured vs Infected (Out of closed cases) in %
    tot_decision = cum_deaths + cum_recov  # Total cases with outcome
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    curve1, = ax1.plot(day[case_1:], cum_deaths[case_1:] * 100 / tot_decision, label="% Dead", color='r')
    curve2, = ax2.plot(day[case_1:], cum_recov[case_1:] * 100 / tot_decision, label="% Recovered", color='b')
    curves = [curve1, curve2]
    ax1.legend(curves, [curve.get_label() for curve in curves])
    plt.xlabel('Day')
    ax1.set_ylabel('% Dead')
    ax2.set_ylabel('% Recovered')
    xticklabels = day[case_1::2]
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels(xticklabels, rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.title('% Dead vs % Recovered - Based on cases with Outcome')
    plt.savefig(output_path + '\\' + 'DeadRecov.png')

    # 8 Case Doubling Rate
    doublingRate(7, cum_cases, 'Case Doubling Rate', state, 8, day, output_path)
    # 9 Death Doubling Rate
    doublingRate(7, cum_deaths, 'Death Doubling Rate', state, 9, day, output_path)

    # 10 Net cases growth
    net_cases = cum_cases - cum_recov - cum_deaths  # Cumulative cases - recovered - dead
    new_netcases = [j - i for i, j in zip(net_cases[:-1], net_cases[1:])]
    new_netcases.insert(0, net_cases.values[0])  # Subtraction eats away first value, copy from cumulative list
    new_netcases = pd.Series(new_netcases)  # Converting list to Series
    dfc.insert(len(dfc.columns), "new_netcases", new_netcases)
    # Alternate - Slightly less elegant
    # new_cases2 = np.diff(cum_cases)
    case_1 = np.argmax(new_netcases > 0)
    new_netcases_roll = new_netcases.rolling(5).mean()
    y_pos = range(len(day))
    plt.figure(10)
    plt.title('Daily new "net" cases' + ' - ' + state)
    plt.bar(day[case_1:], new_netcases[case_1:])
    # Rotation of the bars names
    plt.plot(day[case_1:], new_netcases_roll[case_1:], 'r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Number of new "net" cases / day')
    plt.xticks(day[case_1::2], rotation=90)
    leg1 = '5 day moving average - ' + state
    leg2 = 'Number of new "net = new - recovered" cases / day - ' + state
    plt.legend([leg1, leg2])
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'NewNetCases.png')

    # 11 Cumulative Active cases
    plt.figure(11)
    plt.suptitle('Cumulative Active Cases' + ' - ' + state)
    plt.subplot(2, 1, 1)
    plt.grid(True, which="both")
    plt.semilogy(day[case_1:], net_cases[case_1:])
    plt.ylim([100, max(net_cases) + 1000])
    plt.xlabel('Date')
    plt.ylabel('Cumulative cases (Semilog plot)')
    plt.xticks(day[case_1::2], rotation=90)
    plt.subplot(2, 1, 2)
    plt.grid(True, which="both")
    plt.plot(day[case_1:], net_cases[case_1:])
    plt.ylim([100, max(net_cases) + 1000])
    plt.xlabel('Date')
    plt.ylabel('Cumulative Active cases')
    plt.xticks(day[case_1::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'CumulativeActiveCases.png')


read_file_cases = r'C:\Users\masoo\PycharmProjects\Covid-19\covid_19_india.csv'
read_file_test = r'C:\Users\masoo\PycharmProjects\Covid-19\StatewiseTestingDetails.csv'
df_cases = pd.read_csv(read_file_cases)
unique_states = df_cases['State/UnionTerritory'].unique()
for states in unique_states:
    print(states)
    plt.close('all')
    Covid19(states)

# Todo
# SIR model to estimate peak
# Testing statistics

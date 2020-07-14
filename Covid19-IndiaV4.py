import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DateTime as dt
import os


#
# Data source - https://www.kaggle.com/sudalairajkumar/covid19-in-india

# Function definitions
def doublingRate(di, vector, title, state, figno, day, output_path, ylabel, savename):
    cdr_interval = di  # Case or Death doubling interval
    cum_cases = vector.to_numpy()  # Cumulative cases converted to numpy
    cdr_pts = []  # Doubling rate points
    cdr_idx = np.argmax(cum_cases > 20) + cdr_interval  # n+7 idx after 20 case

    if cum_cases[-1] > 20:
        for cdr in range(cdr_idx, len(cum_cases)):
            cdr_pts.append(cdr_interval / (np.log2(cum_cases[cdr]) - np.log2(cum_cases[cdr - cdr_interval])))
        plt.figure(figno)
        plt.title(title + ' - ' + state)
        plt.plot(day[cdr_idx:], cdr_pts[:], 'o-')
        plt.xlabel('Date')

        plt.ylabel(ylabel)
        plt.xticks(day[cdr_idx::2], rotation=90)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.savefig(output_path + '\\' + savename)


def testingplt(x, y1, y2, title, state, figno, savename, output_path, ylabel):
    plt.figure(figno)
    plt.suptitle(title + ' - ' + state)
    plt.subplot(2, 1, 1)
    plt.bar(x, y1)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.xticks(x[0::2], rotation=90)
    plt.legend(['Daily samples tested'])
    plt.subplot(2, 1, 2)
    plt.bar(x, y2 * 100 / y1)
    plt.xlabel('Day')
    plt.ylabel('% of samples positive')
    plt.xticks(x[0::2], rotation=90)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + savename + '.png')


def SIR(stated, df_casesx, df_testx, N):
    """Here is a brief application of SIR model to predict Beta, Gamma, and eventually R0. Further mean values of
    beta and gamma will be used to estimate the Pandemic peak
    SIR model contains three diff equations
    dSdt = - beta * S(t)/N * I(t)
    dIdt = dSdt - I(t) * gamma
    dRdt = I(t) * gamma """
    N = 19000000
    cum_cases = dfc.Confirmed.rolling(5).mean()
    cum_deaths = dfc['Deaths'].rolling(5).mean()
    cum_recov = dfc.Cured.rolling(5).mean()
    new_cases = DeltaCalc(cum_cases)
    new_recov = DeltaCalc(cum_recov)
    new_deaths = DeltaCalc(cum_deaths)
    case_100 = np.argmax(cum_cases > 100)
    cum_cases = cum_cases[case_100:].values
    cum_deaths = cum_deaths[case_100:].values
    cum_recov = cum_recov[case_100:].values
    new_recov = new_recov[case_100:].values
    new_cases = new_cases[case_100:].values
    gamma = new_recov / (cum_cases - cum_deaths - cum_recov)
    S = N - cum_cases
    beta = new_cases * N / (S * (cum_cases - cum_deaths - cum_recov))
    gamma = gamma.replace([np.nan], 0)
    gamma = gamma[gamma != 0]


def DeltaCalc(vector):
    cum = vector
    delta = [j - i for i, j in zip(cum[:-1], cum[1:])]
    delta.insert(0, cum.values[0])
    delta = pd.Series(delta)
    return delta


def Covid19(stated, df_casesx, df_testx):
    state = stated
    # Update this on your computer to have the correct output path
    output_path = 'C:\\Users\\masoo\\PycharmProjects\\Covid-19\\Results_India\\' + state
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)

    # Filter data-frame for state
    dfc = df_casesx[df_cases["State/UnionTerritory"] == state]
    dft = df_testx[df_testx["State"] == state]

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
    plt.title('Daily new cases detected. This number should be looked in conjunction with samples tested each day' +
              '- ' + state)
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
    plt.savefig(output_path + '\\' + 'AANewCases.png')

    # 2 Cumulative cases
    plt.figure(2)
    plt.suptitle('Number of samples on linear and semi-log graph. Flat semi-log plot is better' + ' - ' + state)
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
    plt.title('Number of new deaths each day. This number is more interesting to look at, compared to the daily new '
              'cases ' + ' - ' + state)
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
    plt.savefig(output_path + '\\' + 'ABNewDeaths.png')

    # 4 Death cumulative
    plt.figure(4)
    plt.suptitle(
        'Numbers of deaths on linear and semi-log graph. Semi-log graph flat = No exponential growth' + ' - ' + state)
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
    plt.savefig(output_path + '\\' + 'ACCumulativeDeaths.png')

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
    plt.title('Daily new Recoveries. The bigger this number, the better. It is also impacted by number of samples '
              'being tested each day' + ' - ' + state)
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
    plt.savefig(output_path + '\\' + 'ADDailyRecovs.png')

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
    plt.savefig(output_path + '\\' + 'AECumRecovs.png')

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
    plt.savefig(output_path + '\\' + 'AFDeadRecov.png')

    # 8 Case Doubling Rate
    doublingRate(7, cum_cases, 'Number of days in which the cases are expected to double. Higher value, better', state,
                 8, day, output_path, 'Case Doubling Rate [days]', 'AGCDR')
    # 9 Death Doubling Rate
    doublingRate(7, cum_deaths, 'Number of days in which the deaths are expected to double. Higher value, better',
                 state, 9, day, output_path, 'Death doubling rate [days]', 'AHDDR')

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
    plt.title('New net cases = Daily new cases - Daily new recoveries. Lower is better' + ' - ' + state)
    plt.bar(day[case_1:], new_netcases[case_1:])
    # Rotation of the bars names
    plt.plot(day[case_1:], new_netcases_roll[case_1:], 'r', linewidth=3)
    plt.xlabel('Date')
    plt.ylabel('Number of new "net" cases / day')
    plt.xticks(day[case_1::2], rotation=90)
    leg1 = '5 day moving average - ' + state
    leg2 = 'Net cases each day - ' + state
    plt.legend([leg1, leg2])
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(output_path + '\\' + 'AINewNetCases.png')

    # 11 Cumulative Active cases
    plt.figure(11)
    plt.suptitle('Number of active samples on linear and semi-log graph. Flat semi log graph indicates slowing pandemic'
                 + ' - ' + state)
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
    plt.savefig(output_path + '\\' + 'AJCumulativeActiveCases.png')

    # 12 Total daily samples tested
    day = []
    for dates in dft.Date:
        dayT = dates[8:10] + '/' + dates[5:7]
        day.append(dayT)
    cum_pos = dft.Positive
    cum_test = dft.TotalSamples
    daily_tests = [j - i for i, j in zip(cum_test[:-1], cum_test[1:])]
    daily_pos = [j - i for i, j in zip(cum_pos[:-1], cum_pos[1:])]
    daily_tests.insert(0, cum_test.values[0])
    daily_tests = pd.Series(daily_tests)
    daily_pos.insert(0, cum_pos.values[0])
    daily_pos = pd.Series(daily_pos)
    # Daily test samples - Stats
    titleD = 'Daily samples tested, and daily % positive. Lower % positive is better'
    testingplt(day, daily_tests, daily_pos, titleD, state, 12, 'AKDailyTestSamples', output_path,
               'Daily samples tested')
    # Cumulative samples - tests
    titleD = 'Cumulative samples tested, and Cumulative % positive. Lower % positive is better'
    testingplt(day, cum_test, cum_pos, titleD, state, 13, 'ALCumulativeTestSamples', output_path,
               'Cumulative samples tested')


read_file_cases = r'C:\Users\masoo\PycharmProjects\Covid-19\covid_19_india.csv'
read_file_test = r'C:\Users\masoo\PycharmProjects\Covid-19\StatewiseTestingDetails.csv'
df_cases = pd.read_csv(read_file_cases)
df_test = pd.read_csv(read_file_test)
unique_states = df_cases['State/UnionTerritory'].unique()

# Sync state names
df_test['State'] = df_test['State'].replace({'Telangana': 'Telengana'})

for states in unique_states:
    print(states)
    plt.close('all')
    if states == 'Unassigned' or states == 'Cases being reassigned to states':
        print('This is invalid state')
    else:
        Covid19(states, df_cases, df_test)

# Todo
# SIR model to estimate peak

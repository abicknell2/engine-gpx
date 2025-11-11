'''uncertain tests based on tp-rampmodels/tp_study'''

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
# NOTE: bootstrapped was written by facebook
from uncertainty_tools.uncertaininputs import (ThreePointVariable, UncertainInput)


def test_uncertainty():

    ## Create uncertainty around the layup rate

    unvar = UncertainInput(skin_processes['ATL tack & preform'].layupRate, ThreePointVariable(7, 10, 20))

    ## Evaluate risk around layup rate

    # substitute for the expected value of the layup rate
    m.substitutions[unvar.var] = unvar.uncertainty.expectation()

    sol_baseline = m.solve()

    unvar.quantify_total_risk(sol_baseline).sum()

    unvar.quantify_total_risk(sol_baseline, measure='net').sum()

    sol_baseline['cost']

    ## Confidence bounds for unit cost (but can be any output)

    # import matplotlib.pyplot as plt

    samples = unvar.uncertainty.sample(size=20)
    samples

    sweep_sol = m.sweep({unvar.var: samples})

    bs_result = bs.bootstrap(sweep_sol['cost'].magnitude, stat_func=bs_stats.mean)
    print(bs_result)

    ### 99% Confidence intervals
    bs.bootstrap(sweep_sol['cost'].magnitude, stat_func=bs_stats.mean, alpha=0.01)

    bs_result.error_fraction() * 100

    bs_result.error_width() * units('USD')

    bs.bootstrap(sweep_sol['sensitivities']['variables'][unvar.var], stat_func=bs_stats.median)

    bs.bootstrap(sweep_sol['variables'][skin_cells['ATL tack & preform'].m].magnitude, stat_func=bs_stats.mean)

    ### Using Built-in Matplotlib box plot

    # plt.axes()
    # result = plt.boxplot(sweep_sol['cost'].magnitude,
    #                      bootstrap=10000,
    #                      labels=['cost'],
    # #                      whis=[5,95],
    #                      notch=False,
    #                     )
    # plt.ylabel('Unit Cost')
    # plt.show()

    # plt.axes()
    # plt.boxplot(sweep_sol['cost'].magnitude,
    #             0, 'rs', 0,
    #             bootstrap=10000,
    #            )
    # plt.show()

    ## The matplotlib bootstrap & boxplot method doesn't seem to work. The boxplot just shows min and max on the whiskers and not really the CI. It's questionable if the quantiles are working either.

    ## Re-run using `pybootstrap`

    ##_Sauce: https://github.com/mvanga/pybootstrap_

    ## Outputs for excel example charts

    ### PDF of the uncertainty (liklihood)

    x = unvar.uncertainty.x
    y = unvar.uncertainty.get_pdf(x)
    for i in range(len(x)):
        print(str(x[i]) + ',' + str(y[i]))

    ### Impact

    y = unvar.quantify_total_risk(sol_baseline, measure='net').magnitude
    for i in range(len(x)):
        print(str(x[i]) + ',' + str(y[i]))

import matplotlib.pyplot as plt
import numpy as np

def plot_continuous_perturb():
    x = [0.25, 0.5, 0.75, 1.00]
    y_vanillagrad = [0.12097, 0.25680, 0.37633, 0.47117]
    y_smoothagrad = [0.12109, 0.25788, 0.37746, 0.47231]
    y_itergrad = [0.12813, 0.27969, 0.41367, 0.53209]
    y_inputgrad = [0.04323, 0.08623, 0.12550, 0.15896]
    #y_refgrad = [0.08143, 0.14912, 0.22320, 0.28824]

    plt.ylabel('prob change')
    plt.xlabel('epsilon')
    plt.plot(x, y_vanillagrad, "--go", label="vanilla")
    plt.plot(x, y_smoothagrad, "--ro", label="smoothGrad")
    plt.plot(x, y_itergrad, "--bo", label="iterGrad")
    plt.plot(x, y_inputgrad, "--co", label="input*Grad")
    #plt.plot(x, y_refgrad, "--yo", label="refGrad")
    plt.legend(loc="upper left")
    plt.show()

def plot_word_2zero():
    x = [1, 2, 3, 4]
    y_vanillagrad = [0.10019287, 0.13370584, 0.1685176, 0.19753554]
    y_smoothagrad = [0.10019287, 0.13358119, 0.16842477, 0.19651913]
    y_itergrad = [0.10524706, 0.13824082, 0.1767163, 0.21206722]
    y_inputgrad = [0.18739992, 0.32716784, 0.4284067, 0.48404338]
    #y_refgrad = [0.12238, 0.17018, 0.21530, 0.25642]

    plt.ylabel('prob change')
    plt.xlabel('num word remove')
    plt.plot(x, y_vanillagrad, "--go", label="vanilla")
    plt.plot(x, y_smoothagrad, "--ro", label="smoothGrad")
    plt.plot(x, y_itergrad, "--bo", label="iterGrad")
    plt.plot(x, y_inputgrad, "--co", label="input*Grad")
    #plt.plot(x, y_refgrad, "--yo", label="refGrad")
    plt.legend(loc="upper left")
    plt.show()


def plot_continuous_perturb_bert():
    x = [0.05, 0.10, 0.15, 0.20]
    y_vanillagrad = [0.09258, 0.20991, 0.30658, 0.37879]
    y_smoothagrad = [0.09259, 0.20992, 0.30663, 0.37881]
    y_itergrad = [0.12411, 0.26307, 0.41530, 0.53245]
    y_inputgrad = [0.01001, 0.03228, 0.05354, 0.072007]
    y_integrad = [0.00744, 0.02771, 0.046487, 0.059286]

    plt.ylabel('prob change')
    plt.xlabel('epsilon')
    plt.plot(x, y_vanillagrad, "--go", label="vanilla")
    plt.plot(x, y_smoothagrad, "--ro", label="smoothGrad")
    plt.plot(x, y_itergrad, "--bo", label="iterGrad")
    plt.plot(x, y_inputgrad, "--co", label="input*Grad")
    plt.plot(x, y_integrad, "--yo", label="integratedGrad")
    plt.legend(loc="upper left")
    plt.show()


def plot_word_2zero_bert():
    x = [1, 2, 3, 4]
    y_vanillagrad = [0.12057786, 0.18185633, 0.22508102, 0.27534258]
    y_smoothagrad = [0.12057786, 0.18185633, 0.22508102, 0.27535298]
    y_itergrad = [0.12409556, 0.18188833, 0.22819398, 0.27837211]
    y_inputgrad = [0.08384714, 0.11675363, 0.14601074, 0.18890988]
    y_integrad = [0.17686895, 0.32740351, 0.39470266, 0.45144566]

    plt.ylabel('prob change')
    plt.xlabel('num word remove')
    plt.plot(x, y_vanillagrad, "--go", label="vanilla")
    plt.plot(x, y_smoothagrad, "--ro", label="smoothGrad")
    plt.plot(x, y_itergrad, "--bo", label="iterGrad")
    plt.plot(x, y_inputgrad, "--co", label="input*Grad")
    plt.plot(x, y_integrad, "--yo", label="integratedGrad")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    # plot_continuous_perturb()
    # plot_word_2zero()
    plot_continuous_perturb_bert()
    # plot_word_2zero_bert()


# 0.27744629605555204
# 0.3220100518315806
# 0.28063909282277905
# 0.06997106305921769

# [0.09242196 0.12971233 0.17555343 0.20424623]
# [0.07321332 0.10952006 0.15268257 0.18200028]
# [0.09198395 0.12923208 0.1781656  0.20788084]
# [0.17996611 0.30893916 0.39811452 0.45716871]
# [0.0892798  0.15383358 0.20348688 0.23439437]

# [0.12736167 0.17625694 0.20216213 0.22481955]
# [0.11576456 0.16266228 0.19637873 0.22164795]
# [0.12864035 0.17693503 0.20433538 0.22707783]
# [0.13166906 0.17936407 0.20444998 0.2206258 ]
# [0.15401147 0.22201061 0.26101835 0.28277046]
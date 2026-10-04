import matplotlib.pyplot as plt
import wfdb


def main():
    # Load one record to inspect the raw ECG signal.
    record = wfdb.rdrecord("data/100")

    print("Sampling frequency:", record.fs, "Hz")
    print("Number of samples:", record.sig_len)
    print("Number of channels:", record.n_sig)

    ecg_signal = record.p_signal[:, 0]
    seconds = 10
    samples = int(seconds * record.fs)

    # Plot a short segment from the first ECG channel.
    plt.figure(figsize=(15, 4))
    plt.plot(ecg_signal[:samples])
    plt.title("MIT-BIH Record 100: First 10 Seconds")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

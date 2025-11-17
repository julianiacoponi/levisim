from levisim.core.constants import π, e


# conversion constants: "divide by this factor to go from..."

# samples
megasamples = 1e6  # samples to megasamples


# angle
degrees = π  / 180  # radians to degrees


# space
m = 1  # metre
cm = 1e-2  # metres to centimetres
mm = 1e-3  # metres to millimetres
μm = 1e-6  # metres to microns
nm = 1e-9  # metres to nanometres
pm = 1e-12  # metres to picometres


# time
s = 1  # second
ms = 1e-3  # seconds to milliseconds
μs = 1e-6  # seconds to microseconds
ns = 1e-9  # seconds to nanoseconds


# rotational frequency
# NOTE: these also just convert "angular" frequency to linear frequency
# i.e. via ω = 2πf, where [ω] = [2π·Hz] i.e. with no radians!
Hz = π + π  # radians/second to hertz
kHz = 1e3 * Hz  # radians/second to kilohertz
MHz = 1e6 * Hz  # radians/second to megahertz
GHz = 1e9 * Hz  # radians/second to gigahertz
THz = 1e12 * Hz  # radians/second to terahertz


# frequency (sad namespace clash)
hertz = 1  # hertz
kiloHz = 1e3  # hertz to kilohertz
megaHz = 1e6  # hertz to megahertz
gigaHz = 1e9  # hertz to gigahertz
teraHz = 1e12  # hertz to terahertz


# mass
kg = 1  # kilogram
grams = 1e-3 * kg  # kilograms to grams (namespace clash with gravitational g)
ng = 1e-9 * grams  # kilograms to nanograms
pg = 1e-12 * grams  # kilograms to picograms
fg = 1e-15 * grams  # kilograms to femtograms


# energy
eV = e  # joules to electronvolts
keV = 1e3 * eV  # joules to kiloelectronvolts
MeV = 1e6 * eV  # joules to megaelectronvolts


# power
W = 1  # watt
mW = 1e-3  # watts to milliwatts


# force
N = 1  # newton
fN = 1e-15  # newtons to femtonewtons
aN = 1e-18  # newtons to attonewtons
zN = 1e-21  # newtons to zeptonewtons
yN = 1e-24  # newtons to yoctonewtons


# pressure
bar = 100_000  # pascals to bar
mbar = 1e-3 * bar  # pascals to millibar
torr = 133.3224  # pascals to torr, roughly 4/3 mbar!


# temperature
K = 1  # kelvin
mK = 1e-3  # kelvin to millikelvin


# voltage
V = 1  # volt
mV = 1e-3  # volts to millivolts


# current
A = 1
mA = 1e-3  # amps to milliamps
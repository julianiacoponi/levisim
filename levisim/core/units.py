from levisim.core.constants import π, e


# conversion constants: "divide by this factor to go from..."

# samples
Megasamples = 1e6  # samples to Megasamples


# space
cm = 1e-2  # metres to centimetres
mm = 1e-3  # metres to millimetres
μm = 1e-6  # metres to microns
nm = 1e-9  # metres to nanometres
pm = 1e-12 # metres to picometres


# time
ms = 1e-3  # seconds to milliseconds
μs = 1e-6  # seconds to microseconds
ns = 1e-9  # seconds to nanoseconds


# angular
degrees = π  / 180  # radians to degrees
Hz = π + π  # radians/second to Hertz
kHz = 1e3 * Hz  # radians/second to kiloHertz
MHz = 1e6 * Hz  # radians/second to MegaHertz
GHz = 1e9 * Hz  # radians/second to GigaHertz
THz = 1e12 * Hz  # radians/second to TeraHertz


# frequency (sad namespace clash)
kiloHz = 1e3  # Hertz to kiloHertz
MegaHz = 1e6  # Hertz to MegaHertz
GigaHz = 1e9  # Hertz to GigaHertz
TeraHz = 1e12  # Hertz to TeraHertz


# mass
ng = 1e-9 * 1e-3 # kilograms to femtograms
pg = 1e-12 * 1e-3 # kilograms to femtograms
fg = 1e-15 * 1e-3 # kilograms to femtograms


# energy
eV = e  # Joules to electron-Volts
keV = 1e3 * eV  # Joules to kilo-electron-Volts
MeV = 1e6 * eV  # Joules to Mega-electron-Volts


# power
mW = 1e-3  # Watts to milliWatts


# force
fN = 1e-15  # Newtons to femtoNewtons
aN = 1e-18  # Newtons to attoNewtons
zN = 1e-21  # Newtons to zeptoNewtons
yN = 1e-24  # Newtons to yoctoNewtons


# pressure
bar = 100_000  # Pascals to bar
mbar = 1e-3 * bar  # Pascals to millibar
Torr = 133.3224  # Pascals to Torr, roughly 4/3 mbar!


# temperature
K = 1  # Kelvin
mK = 1e-3  # Kelvin to milliKelvin

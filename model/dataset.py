import json
import random

# Base FSM patterns
fsm_templates = []

# Simple toggle patterns
toggle_patterns = [
    ("A light switch toggles between on and off states", "states: off, on | transitions: off --toggle--> on, on --toggle--> off"),
    ("A door lock toggles between locked and unlocked", "states: locked, unlocked | transitions: locked --key--> unlocked, unlocked --key--> locked"),
    ("A power button switches device on and off", "states: off, on | transitions: off --press--> on, on --press--> off"),
    ("A lamp switches between bright and dim modes", "states: bright, dim | transitions: bright --adjust--> dim, dim --adjust--> bright"),
]

# Multi-state sequential patterns
sequential_patterns = [
    ("A traffic light switches from red to green to yellow in order", "states: red, green, yellow | transitions: red -> green, green -> yellow, yellow -> red"),
    ("A washing machine cycles through wash rinse and spin", "states: idle, wash, rinse, spin | transitions: idle -> wash, wash -> rinse, rinse -> spin, spin -> idle"),
    ("A dishwasher runs through pre-wash main wash rinse and dry cycles", "states: idle, prewash, mainwash, rinse, dry | transitions: idle -> prewash, prewash -> mainwash, mainwash -> rinse, rinse -> dry, dry -> idle"),
    ("A microwave starts heating stops when timer ends", "states: idle, heating, complete | transitions: idle --start--> heating, heating --timer--> complete, complete -> idle"),
]

# Fan/speed control patterns
speed_patterns = [
    ("A fan has three speeds controlled by button", "states: off, low, medium, high | transitions: off -> low, low -> medium, medium -> high, high -> off"),
    ("A ceiling fan cycles through off low medium high speeds", "states: off, low, medium, high | transitions: off -> low, low -> medium, medium -> high, high -> off"),
    ("A blender operates at low medium and high speeds with off state", "states: off, low, medium, high | transitions: off -> low, low -> medium, medium -> high, high -> off"),
    ("An air conditioner has fan modes off low medium high", "states: off, low, medium, high | transitions: off -> low, low -> medium, medium -> high, high -> off"),
]

# Door/gate patterns
door_patterns = [
    ("A door bell rings when button is pressed", "states: idle, ringing | transitions: idle --press--> ringing, ringing --timeout--> idle"),
    ("A garage door opener toggles between closed and open", "states: closed, opening, open, closing | transitions: closed --button--> opening, opening -> open, open --button--> closing, closing -> closed"),
    ("An automatic door opens when sensor detects person", "states: closed, open | transitions: closed --detect--> open, open --timeout--> closed"),
    ("A gate opens when vehicle approaches and closes after passing", "states: closed, open | transitions: closed --approach--> open, open --pass--> closed"),
]

# Elevator patterns
elevator_patterns = [
    ("An elevator moves between floors", "states: floor1, floor2, floor3 | transitions: floor1 --up--> floor2, floor2 --up--> floor3, floor3 --down--> floor2, floor2 --down--> floor1"),
    ("A lift operates between ground first and second floors", "states: ground, first, second | transitions: ground --up--> first, first --up--> second, second --down--> first, first --down--> ground"),
    ("An elevator services basement ground and upper floors", "states: basement, ground, upper | transitions: basement --up--> ground, ground --up--> upper, upper --down--> ground, ground --down--> basement"),
]

# Vending machine patterns
vending_patterns = [
    ("A vending machine gives output after receiving 10 rupees", "states: s0, s5, s10 | transitions: s0 --5rs--> s5, s5 --5rs--> s10, s10 --dispense--> s0"),
    ("A coin slot accepts coins until 20 cents then dispenses", "states: s0, s10, s20 | transitions: s0 --10c--> s10, s10 --10c--> s20, s20 --dispense--> s0"),
    ("A ticket machine collects 50 rupees then prints ticket", "states: s0, s10, s20, s30, s40, s50 | transitions: s0 --10rs--> s10, s10 --10rs--> s20, s20 --10rs--> s30, s30 --10rs--> s40, s40 --10rs--> s50, s50 --print--> s0"),
]

# Turnstile patterns
turnstile_patterns = [
    ("A turnstile unlocks on token and locks again after passing", "states: locked, unlocked | transitions: locked --token--> unlocked, unlocked --pass--> locked"),
    ("A subway gate opens with ticket and closes after entry", "states: closed, open | transitions: closed --ticket--> open, open --entry--> closed"),
    ("A parking barrier lifts on payment and lowers after car passes", "states: down, up | transitions: down --pay--> up, up --pass--> down"),
]

# Media player patterns
player_patterns = [
    ("A music player cycles through stopped playing and paused states", "states: stopped, playing, paused | transitions: stopped --play--> playing, playing --pause--> paused, paused --resume--> playing, playing --stop--> stopped"),
    ("A video player has play pause and stop controls", "states: stopped, playing, paused | transitions: stopped --play--> playing, playing --pause--> paused, paused --play--> playing, playing --stop--> stopped"),
]

# Thermostat patterns
thermostat_patterns = [
    ("A thermostat switches heating on when cold and off when warm", "states: heateroff, heateron | transitions: heateroff --cold--> heateron, heateron --warm--> heateroff"),
    ("A temperature controller activates cooling when hot", "states: coolingoff, coolingon | transitions: coolingoff --hot--> coolingon, coolingon --cool--> coolingoff"),
    ("An AC unit turns on at 25C and off at 20C", "states: off, on | transitions: off --hot--> on, on --cool--> off"),
]

# Battery/charging patterns
battery_patterns = [
    ("A battery charger indicates charging full and disconnected states", "states: disconnected, charging, full | transitions: disconnected --plug--> charging, charging --complete--> full, full --unplug--> disconnected"),
    ("A phone charging status shows not charging charging and charged", "states: notcharging, charging, charged | transitions: notcharging --connect--> charging, charging --full--> charged, charged --disconnect--> notcharging"),
]

# Alarm patterns
alarm_patterns = [
    ("A burglar alarm is armed detects intrusion sounds alarm then resets", "states: armed, intrusion, alarmsounding, reset | transitions: armed --detect--> intrusion, intrusion -> alarmsounding, alarmsounding --reset--> armed"),
    ("A fire alarm monitors detects smoke triggers alarm and resets", "states: monitoring, smokedetected, alarmtriggered | transitions: monitoring --smoke--> smokedetected, smokedetected -> alarmtriggered, alarmtriggered --reset--> monitoring"),
]

# Sensor patterns
sensor_patterns = [
    ("A motion sensor light turns on with movement and off after timeout", "states: off, on | transitions: off --motion--> on, on --timeout--> off"),
    ("A proximity sensor activates when object is near", "states: inactive, active | transitions: inactive --detect--> active, active --clear--> inactive"),
    ("A light sensor turns lights on at night and off during day", "states: lightsoff, lightson | transitions: lightsoff --dark--> lightson, lightson --bright--> lightsoff"),
]

# ATM patterns
atm_patterns = [
    ("An ATM transitions from idle to card inserted to PIN entry to transaction", "states: idle, cardinserted, pinentry, transaction | transitions: idle --insert--> cardinserted, cardinserted --enterpin--> pinentry, pinentry --correct--> transaction, transaction -> idle"),
]

# Combine all patterns
all_patterns = (
    toggle_patterns + sequential_patterns + speed_patterns + 
    door_patterns + elevator_patterns + vending_patterns +
    turnstile_patterns + player_patterns + thermostat_patterns +
    battery_patterns + alarm_patterns + sensor_patterns + atm_patterns
)

# Create variations with paraphrasing
dataset = []
for input_text, output_text in all_patterns:
    dataset.append({"input": input_text, "output": output_text})

# Add more variations by slight modifications
variations = [
    ("A switch toggles between two states", "states: state1, state2 | transitions: state1 --toggle--> state2, state2 --toggle--> state1"),
    ("A button press changes system state", "states: off, on | transitions: off --press--> on, on --press--> off"),
    ("A coffee machine dispenses after coin insertion", "states: idle, coininserted, dispensing | transitions: idle --coin--> coininserted, coininserted -> dispensing, dispensing -> idle"),
    ("A toaster starts toasting on lever press and pops up when done", "states: idle, toasting, popup | transitions: idle --press--> toasting, toasting --done--> popup, popup -> idle"),
    ("A printer cycles through ready printing and out of paper states", "states: ready, printing, outofpaper | transitions: ready --print--> printing, printing --nopaper--> outofpaper, outofpaper --refill--> ready"),
    ("A computer boots up runs sleeps and shuts down", "states: off, booting, running, sleep, shuttingdown | transitions: off -> booting, booting -> running, running --sleep--> sleep, sleep --wake--> running, running --shutdown--> shuttingdown, shuttingdown -> off"),
]

dataset.extend([{"input": inp, "output": out} for inp, out in variations])

# Save dataset
with open('fsm_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"✓ Generated dataset with {len(dataset)} samples")
print(f"✓ Saved to fsm_dataset.json")
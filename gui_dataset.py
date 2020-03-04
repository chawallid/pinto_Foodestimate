import cv2
import os
import shutil
import csv

from datetime import date

from threading import Thread
import tkinter
from tkinter import ttk
from tkinter import messagebox

import time
import sys

import RPi.GPIO as GPIO

sys.path.insert(0, "c:/python37/lib/site-packages/")

EMULATE_HX711 = False
cap = None
referenceUnit = 1150
referenceUnit2 = 1000
referenceUnit3 = 1300
i = 1
if not EMULATE_HX711:
    from hx711 import HX711
else:
    from emulated_hx711 import HX711
# Set pin 10 to be an input pin and set initial value to be pulled low (off)
# #######(dout,sdk)
# hx = hx2 = hx3 = None
hx = HX711(20, 21)  # sensor  mid
hx2 = HX711(6, 13)  # sensor right
hx3 = HX711(19, 26)  # sensor left
buttonPin = 22  # pin 15

record_thread = None

is_capture = None
start = False

root = None

rgb_saving_frames = []
depth_saving_frames = []

# -------begin capturing and saving video
def startrecording(saving_dir):
    global start
    start = True

    global datasets

    counter = {}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("w,h,fps", w, h, fps)

    while start:
        ret, im = cap.read()
        #         resized_im = cv2.resize(im,(640,360))
        cv2.imshow("camera", im)
        button_state = GPIO.input(buttonPin)

        val_mid = max(0, int(hx.get_weight(5)))
        val_right = max(0, int(hx2.get_weight(5)))
        val_left = max(0, int(hx3.get_weight(5)))
        # print(val_left,val_mid,val_right)
        # print('buttonState',button_state)

        if cv2.waitKey(1) & button_state == 1:  # GPIO.HIGH
            print("start sent data.....")

            dmy = date.today().strftime("%Y_%m_%d")
            dataset_name = f"{dmy}"  # "-{dataset_name_entry.get()}"
            current_radio_name = get_radio_name(menu_choice.get())
            current_menu_name = datasets[menu_choice.get()]["name_txt"].get()
            if current_menu_name == "":
                print("BE CAREFUL: empty menu name!!")
                continue

            csv_dataset_path = os.path.join(
                saving_dir, f"{dataset_name}-{current_menu_name}.csv"
            )

            if current_radio_name not in counter:
                counter[current_radio_name] = 1
            filename = (
                f"{dmy}-{current_menu_name}-{counter[current_radio_name]}.jpg"
            )
            counter[current_radio_name] += 1
            sample = {
                "filename": filename,
                "left_sensor": val_left,
                "top_sensor": val_mid,
                "right_sensor": val_right,
            }
            sample["position"] = current_radio_name
            for index, dataset in enumerate(datasets):
                radio_name = get_radio_name(index)
                menu_name = dataset["name_txt"].get()
                sample[f"{radio_name}_name"] = menu_name

                total_value = 0
                for menu in dataset["ingrs"]:
                    ingr_name = menu["name_txt"].get()
                    if ingr_name == "":
                        continue
                    total_weight = (
                        int(menu["total_weight_txt"].get())
                        if menu["total_weight_txt"].get() != ""
                        else 0
                    )
                    current_weight = (
                        int(menu["current_weight_txt"].get())
                        if menu["current_weight_txt"].get() != ""
                        else 0
                    )
                    weight = total_weight - current_weight
                    if index == menu_choice.get():
                        sample[f"{ingr_name}"] = weight  # f"{radio_name}_"
                    total_value += weight
                sample[f"{radio_name}_total"] = total_value

            total_value = 0
            #             total_value += int(food_weight_entry_1.get()) if food_weight_entry_1.get() != "" else 0
            #             sample["total"] = total_value
            #             print(sample)
            sample = dict([(k, v) for k, v in sample.items() if v != ""])
            print("===")
            print(sample)

            file_exists = os.path.isfile(csv_dataset_path)
            with open(csv_dataset_path, "a") as f:
                writer = csv.DictWriter(f, sample.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(sample)

            #             cv2.resize(im,(640,480))
            im_saving_dir = os.path.join(
                saving_dir, f"{dataset_name}-{current_menu_name}"
            )
            if not os.path.exists(im_saving_dir):
                os.makedirs(im_saving_dir)
            im_saving_path = os.path.join(im_saving_dir, filename)
            cv2.imwrite(im_saving_path, im)
            print(f"saved {im_saving_path}")
        # print(val_left,val_mid,val_right)
        hx.power_down()
        hx2.power_down()
        hx3.power_down()

        hx.power_up()
        hx2.power_up()
        hx3.power_up()

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    start = False


def start_recording_proc(saving_dir):
    #     if os.path.exists(saving_dir):
    #         raise OSError(f"File path {saving_dir} is existed.")

    global record_thread
    record_thread = Thread(target=startrecording, args=(saving_dir,))
    record_thread.start()

    # p.start()


# -------end video capture and stop tk
def stop_recording():
    msg_box = messagebox.askquestion(
        "Exit Application",
        "Are you sure you want to exit the application",
        icon="warning",
        default="no",
    )
    if msg_box == "yes":
        exit_program()


def exit_program():
    global start
    start = False

    global record_thread
    if record_thread is not None:
        record_thread.join()
        record_thread = None

    # e.set()

    global root
    root.quit()
    root.destroy()


def capture():
    global is_capture
    is_capture = True
    print("CAPTURE", is_capture)


def keypress_capture(e):
    capture()


def revert():
    global rgb_saving_frames
    global depth_saving_frames

    if len(rgb_saving_frames) > 0 and len(depth_saving_frames) > 0:
        rgb_saving_frames.pop()
        depth_saving_frames.pop()

    print(len(rgb_saving_frames), len(depth_saving_frames))


def keypress_revert(e):
    revert()


def testVal(inStr, acttyp):
    if acttyp == "1":  # insert
        if not inStr.isdigit():
            return False
    return True


def get_radio_name(index):
    if index == 0:
        return "left"
    elif index == 1:
        return "top"
    elif index == 2:
        return "right"
    return f"Menu {index}"


def draw_dataset_gui(root):
    # SAVING_DIR = "/home/tommie/dev/food-scanner/datasets/"
    SAVING_DIR = (
        "/home/pi/Downloads/HX711-master/HX711_Python3/hx711py/datasets/"
    )

    total, used, free = shutil.disk_usage(SAVING_DIR)
    free_space_gib = free // (2 ** 30)
    #     if free_space_gib < 1:
    #         raise OSError("Dangerous: disk may have not enough spaces")

    startbutton = tkinter.Button(
        root,
        width=10,
        height=1,
        text="START",
        command=lambda: start_recording_proc(SAVING_DIR),
    )
    stopbutton = tkinter.Button(
        root, width=10, height=1, text="STOP", command=stop_recording
    )
    capturebutton = tkinter.Button(
        root, width=10, height=1, text="CAPTURE", command=capture
    )
    global dataset_name_entry
    dataset_name_entry = tkinter.Entry(root)

    global menu_choice
    menu_choice = tkinter.IntVar(root, 0)
    global datasets
    datasets = []
    NUM_DATASET = 3
    for dataset_index in range(NUM_DATASET):
        dataset = {}
        radio_name = get_radio_name(dataset_index)
        dataset["radio"] = tkinter.Radiobutton(
            root,
            text=radio_name,
            padx=20,
            variable=menu_choice,
            value=dataset_index,
        )
        dataset["name_lbl"] = tkinter.Label(root, text="Menu Name:")
        dataset["name_txt"] = tkinter.Entry(root)
        dataset["ingrs"] = []
        for ingr_index in range(1, 4 + 1):
            ingr = {}
            ingr["name_lbl"] = tkinter.Label(root, text=f"Ingr {ingr_index}:")
            ingr["name_txt"] = tkinter.Entry(root, width=10)
            ingr["total_weight_txt"] = tkinter.Entry(
                root, width=8, validate="key"
            )
            ingr["total_weight_txt"]["validatecommand"] = (
                ingr["total_weight_txt"].register(testVal),
                "%P",
                "%d",
            )
            ingr["current_weight_txt"] = tkinter.Entry(
                root, width=8, validate="key"
            )
            ingr["current_weight_txt"]["validatecommand"] = (
                ingr["current_weight_txt"].register(testVal),
                "%P",
                "%d",
            )
            dataset["ingrs"].append(ingr)
        datasets.append(dataset)
    # root.geometry(f"{280*NUM_DATASET}x240+0+0")

    """
    Adjusting layout
    """

    SET_COLUMN_WIDTH = 5

    startbutton.grid(row=0, columnspan=NUM_DATASET * SET_COLUMN_WIDTH)
    #     capturebutton.grid(row=1)
    dataset_name_entry.grid(row=1, columnspan=NUM_DATASET * SET_COLUMN_WIDTH)

    for column_index, dataset in enumerate(datasets):
        row = 3
        dataset["radio"].grid(
            row=row,
            column=0 + (column_index * SET_COLUMN_WIDTH),
            columnspan=SET_COLUMN_WIDTH - 1,
        )
        row += 1
        dataset["name_lbl"].grid(
            row=row, column=0 + (column_index * SET_COLUMN_WIDTH)
        )
        dataset["name_txt"].grid(
            row=row, column=1 + (column_index * SET_COLUMN_WIDTH), columnspan=3
        )
        row += 1
        tkinter.Label(root, text="Name").grid(
            row=row, column=1 + (column_index * SET_COLUMN_WIDTH)
        )
        tkinter.Label(root, text="Total").grid(
            row=row, column=2 + (column_index * SET_COLUMN_WIDTH)
        )
        tkinter.Label(root, text="Current").grid(
            row=row, column=3 + (column_index * SET_COLUMN_WIDTH)
        )
        row += 1
        for ingr in dataset["ingrs"]:
            ingr["name_lbl"].grid(
                row=row, column=0 + (column_index * SET_COLUMN_WIDTH)
            )
            ingr["name_txt"].grid(
                row=row, column=1 + (column_index * SET_COLUMN_WIDTH)
            )
            ingr["total_weight_txt"].grid(
                row=row, column=2 + (column_index * SET_COLUMN_WIDTH)
            )
            ingr["current_weight_txt"].grid(
                row=row, column=3 + (column_index * SET_COLUMN_WIDTH)
            )
            row += 1
        ttk.Separator(root, orient="vertical").grid(
            column=4 + (column_index * SET_COLUMN_WIDTH),
            row=1,
            rowspan=9,
            sticky="ns",
        )

    stopbutton.grid(row=50, columnspan=NUM_DATASET * SET_COLUMN_WIDTH)

    root.bind("<Return>", keypress_capture)
    root.bind("<BackSpace>", keypress_revert)


def main():
    SAVING_DIR = (
        "/home/pi/Downloads/HX711-master/HX711_Python3/hx711py/datasets/"
    )

    total, used, free = shutil.disk_usage(SAVING_DIR)
    free_space_gib = free // (2 ** 30)
    #     if free_space_gib < 1:
    #         raise OSError("Dangerous: disk may have not enough spaces")

    global root
    root = tkinter.Tk()
    root.geometry("200x220+0+0")

    draw_dataset_gui(root)

    # -------begin
    root.mainloop()


def setup_load_cells():
    ######### set switch ##########333

    global EMULATE_HX711

    global cap
    cap = cv2.VideoCapture(0)

    def cleanAndExit():
        print("Cleaning...")

        if not EMULATE_HX711:
            GPIO.cleanup()

        print("Bye!")
        sys.exit()

    GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    hx.set_reading_format("MSB", "MSB")
    hx2.set_reading_format("MSB", "MSB")
    hx3.set_reading_format("MSB", "MSB")

    ################# Calibate ############
    hx.set_reference_unit(referenceUnit)
    hx2.set_reference_unit(referenceUnit2)
    hx3.set_reference_unit(referenceUnit3)

    hx.reset()
    hx2.reset()
    hx3.reset()

    hx.tare()
    hx2.tare()
    hx3.tare()

    print("Tare done! Add weight now...")


if __name__ == "__main__":
    setup_load_cells()
    main()

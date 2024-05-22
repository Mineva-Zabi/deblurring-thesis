def resizeAllImagesInDir(pathToDir):
    import os
    import cv2

    target_width = 1280
    target_height = 720

    for filename in os.listdir(pathToDir):
        abs_filename = os.path.join(pathToDir, filename)
        if os.path.isfile(abs_filename):
            img = cv2.imread(abs_filename)
            if img is not None:
                # Resize to the target dimensions
                img = cv2.resize(img, (target_width, target_height))

                # Ensure the dimensions are multiples of 8
                new_width = target_width - (target_width % 8)
                new_height = target_height - (target_height % 8)
                img = cv2.resize(img, (new_width, new_height))

                cv2.imwrite(abs_filename, img)
            else:
                # Remove invalid or unreadable images
                os.remove(abs_filename)

def blur_images_in_directory(input_directory, output_directory, kernel_size=(50, 50)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        print(input_directory)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            print(filename)
            input_path = os.path.join(input_directory, filename)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Could not read the image \
                at '{input_path}'.")
                return False
            blurred_img = cv2.blur(img, kernel_size)
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, blurred_img)

            print(f"Blurred: {filename}")


def apply_motion_blur(input_directory, output_directory, kernel_size=15):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        print(input_directory)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            print(filename)
            input_path = os.path.join(input_directory, filename)
            img = cv2.imread(input_path)
            if img is None:
                print(f"not read the image at '{input_path}'.")
                return False
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size

            motion_blur = cv2.filter2D(img, -1, kernel)
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, motion_blur)

            print(f"Blurred: {filename}")

def calculate_metrics(input_dir, output_dir):
    psnr_values = []
    ssim_values = []
    input_images = sorted(os.listdir(input_dir))
    output_images = sorted(os.listdir(output_dir))
    
    if not input_images or not output_images:
        print("Input or output directory is empty. \
        Please check the directories.")
        return None, None
    
    for in_img, out_img in zip(input_images, output_images):
        try:
            in_image = imread(os.path.join(input_dir, in_img))
            out_image = imread(os.path.join(output_dir, out_img))
            
            psnr_values.append(psnr(in_image, out_image))
            ssim_value = ssim(in_image, out_image, channel_axis=-1)
            ssim_values.append(ssim_value)
        except Exception as e:
            print(f"Error reading images {in_img} and {out_img}: {e}")
            continue
    
    if not psnr_values or not ssim_values:
        print("No valid image pairs found. \
        Please check the directories and image files.")
        return None, None
    
    return np.mean(psnr_values), np.mean(ssim_values)
    
def deblur_iteration(input_dir, output_dir, task='Deblurring', threshold=10):
    command = f'cd /kaggle/working/MPRNet && \
    python demo.py --task {task} --input_dir {input_dir} \
    --result_dir {output_dir}'
    subprocess.run(command, shell=True)

    initial_psnr, initial_ssim = calculate_metrics(input_dir, output_dir)

    if initial_psnr is None or initial_ssim is None:
        print("Unable to calculate initial metrics. Exiting.")
        return

    print(f'Initial PSNR: {initial_psnr}, \
    Initial SSIM: {initial_ssim}')

    iteration = 1
    while True:
        iteration += 1

        # Run the deblurring command
        command = f'cd /kaggle/working/MPRNet && \
        python demo.py --task {task} --input_dir {output_dir} \
        --result_dir {output_dir}'
        subprocess.run(command, shell=True)

        current_psnr, current_ssim = calculate_metrics(input_dir, output_dir)

        if current_psnr is None or current_ssim is None:
            print("Unable to calculate current metrics. Exiting.")
            break

        print(f'Iteration {iteration}: PSNR: {current_psnr}, \
        SSIM: {current_ssim}')

        psnr_drop = (initial_psnr - current_psnr) / initial_psnr * 100
        ssim_drop = (initial_ssim - current_ssim) / initial_ssim * 100

        print(f'PSNR Drop: {psnr_drop}%, SSIM Drop: {ssim_drop}%')

        if psnr_drop >= threshold or ssim_drop >= threshold:
            print(f'Metrics dropped by {threshold}% at iteration \
            {iteration}. PSNR Drop: {psnr_drop}%, SSIM Drop: {ssim_drop}%')
            break

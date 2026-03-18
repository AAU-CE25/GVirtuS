```bash
wg38up@student.aau.dk@es-dpu-02:~$ lspci | grep -i mellanox
02:00.0 Ethernet controller: Mellanox Technologies MT2894 Family [ConnectX-6 Lx]
02:00.1 Ethernet controller: Mellanox Technologies MT2894 Family [ConnectX-6 Lx]
82:00.0 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.1 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.2 DMA controller: Mellanox Technologies MT43244 BlueField-3 SoC Management Interface (rev 01)
wg38up@student.aau.dk@es-dpu-02:~$ lspci | grep -i bluefield
82:00.0 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.1 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.2 DMA controller: Mellanox Technologies MT43244 BlueField-3 SoC Management Interface (rev 01)
```


GIT HUB CONNECT from the REMOTE
wg38up@student.aau.dk@es-dpu-02:~$ mkdir -p ~/.ssh
wg38up@student.aau.dk@es-dpu-02:~$ chmod 700 ~/.ssh
wg38up@student.aau.dk@es-dpu-02:~$ ssh-keygen -t ed25519 -C "wg38up@student.aau.dk"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/student.aau.dk/wg38up/.ssh/id_ed25519): aau_github_key
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in aau_github_key
Your public key has been saved in aau_github_key.pub
wg38up@student.aau.dk@es-dpu-02:~$ cd ~/.ssh
wg38up@student.aau.dk@es-dpu-02:~$ mv aau_github_key aau_github_key.pub ~/.ssh/
wg38up@student.aau.dk@es-dpu-02:~$ cd ~/.ssh
wg38up@student.aau.dk@es-dpu-02:~/.ssh$ ls
aau_github_key  aau_github_key.pub  config
wg38up@student.aau.dk@es-dpu-02:~$ nano ~/.ssh/config
wg38up@student.aau.dk@es-dpu-02:~$ cat ~/.ssh/config
Host github.com
    IdentityFile ~/.ssh/aau_github_key
wg38up@student.aau.dk@es-dpu-02:~/.ssh$ ssh -T git@github.com
The authenticity of host 'github.com (140.82.121.4)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Hi AAUKajetan! You've successfully authenticated, but GitHub does not provide shell access.
wg38up@student.aau.dk@es-dpu-02:~/.ssh$ cd ..
wg38up@student.aau.dk@es-dpu-02:~$ git clone git@github.com:AAU-CE25/GVirtuS.git
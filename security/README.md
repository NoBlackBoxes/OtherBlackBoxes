# boxes : security

## Notes

### XSS

Cross-site-scripting. When user input is reflected back in delivered page content (without being cleaned), it becomes possible to run arbitrary code by sending sneaky user input.

```html
<script>alert(1)</script>
```

### SQL injection
Add ' to end of URL

### Forbidden Access

Is there another way to access restructed pages?

### Button
inject code into click responses...



### Hideen fields

### Login page...password cracking

rockyoulist.txt

hydra -vV -t 16 -S -L rockyou.txt -p "idk" 34.208.128.127 http-post-form "/479d9e89892c5b613157afcade304547/login:username=^USER^&password=^PASS^:Invalid username"

SWEET
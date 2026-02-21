# Setup and Configuration - Complete Guide

This document provides complete step-by-step instructions for setting up the Executive Assistant automation system from scratch.

## Prerequisites

Before starting, ensure you have:

- A virtual machine (VM) running Ubuntu 22.04 or similar Linux distribution
- SSH access to the VM
- Node.js and npm installed
- API credentials for Monday.com, Asana, and Slack

## Part 1: Install OpenClaw

### Step 1: Install OpenClaw

SSH into your VM and install OpenClaw globally:

```bash
sudo npm install -g openclaw
```

### Step 2: Initialize OpenClaw

Start OpenClaw for the first time to initialize the configuration:

```bash
openclaw gateway
```

This will create the `~/.openclaw` directory and configuration files. You can stop it with `Ctrl+C` after initialization.

### Step 3: Configure WhatsApp

Pair OpenClaw with your WhatsApp account. When you run OpenClaw, it will display a QR code. Scan this QR code with your WhatsApp mobile app:

1. Open WhatsApp on your phone
2. Go to Settings → Linked Devices
3. Tap "Link a Device"
4. Scan the QR code displayed in the terminal

### Step 4: Configure OpenClaw Settings

Edit the OpenClaw configuration file:

```bash
nano ~/.openclaw/openclaw.json
```

Update it with the following configuration (replace the phone number with your WhatsApp number):

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "openai/gpt-4o"
      },
      "models": {
        "openai/gpt-5.1-codex": {
          "alias": "GPT"
        },
        "openai/gpt-4o": {}
      },
      "workspace": "/home/ubuntu/.openclaw/workspace",
      "compaction": {
        "mode": "safeguard"
      },
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8
      }
    }
  },
  "messages": {
    "ackReactionScope": "group-mentions"
  },
  "commands": {
    "native": "auto",
    "nativeSkills": "auto"
  },
  "channels": {
    "whatsapp": {
      "dmPolicy": "allowlist",
      "selfChatMode": false,
      "allowFrom": [
        "+1234567890"
      ],
      "groupPolicy": "allowlist",
      "debounceMs": 0,
      "mediaMaxMb": 50
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "YOUR_GATEWAY_TOKEN_HERE"
    },
    "tailscale": {
      "mode": "off",
      "resetOnExit": false
    },
    "nodes": {
      "denyCommands": [
        "camera.snap",
        "camera.clip",
        "screen.record",
        "calendar.add",
        "contacts.add",
        "reminders.add"
      ]
    }
  },
  "plugins": {
    "entries": {
      "whatsapp": {
        "enabled": true
      }
    }
  },
  "meta": {
    "lastTouchedVersion": "2026.2.17",
    "lastTouchedAt": "2026-02-21T06:10:00.000Z"
  }
}
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

## Part 2: Create API Integration Scripts

### Step 1: Create the Project Directory

```bash
mkdir -p /home/ubuntu/moltbot-actions
cd /home/ubuntu/moltbot-actions
```

### Step 2: Create Monday.com API Script

Create `monday-api.js`:

```bash
nano monday-api.js
```

Paste the following content:

```javascript
const https = require('https');

const MONDAY_API_TOKEN = process.env.MONDAY_API_TOKEN;
const BOARD_ID = 'YOUR_MONDAY_BOARD_ID'; // Sales Pipeline board ID

function createDeal(company, dealValue, stage) {
  const query = `
    mutation {
      create_item (
        board_id: ${BOARD_ID},
        item_name: "${company}",
        column_values: "{\\"numbers\\":\\"${dealValue}\\",\\"status\\":\\"${stage}\\"}"
      ) {
        id
      }
    }
  `;

  const data = JSON.stringify({ query });

  const options = {
    hostname: 'api.monday.com',
    port: 443,
    path: '/v2',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': MONDAY_API_TOKEN,
      'Content-Length': Buffer.byteLength(data)
    }
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        if (res.statusCode === 200) {
          resolve('Deal created in Monday.com');
        } else {
          reject(new Error('Monday.com API error: ' + res.statusCode + ' - ' + body));
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

const company = process.argv[2] || 'Test Company';
const dealValue = process.argv[3] || '50000';
const stage = process.argv[4] || 'Closed Win';

createDeal(company, dealValue, stage)
  .then(result => {
    console.log('✅ ' + result);
    console.log('   Company: ' + company);
    console.log('   Deal Value: $' + dealValue);
    console.log('   Stage: ' + stage);
    process.exit(0);
  })
  .catch(error => {
    console.error('❌ Error: ' + error.message);
    process.exit(1);
  });
```

Save and exit.

### Step 3: Create Asana API Script

Create `asana-api.js`:

```bash
nano asana-api.js
```

Paste the following content:

```javascript
const https = require('https');

const ASANA_API_TOKEN = process.env.ASANA_API_TOKEN;
const WORKSPACE_GID = 'YOUR_ASANA_WORKSPACE_ID';
const PROJECT_TEMPLATE_GID = 'YOUR_ASANA_PROJECT_TEMPLATE_ID';

function createProject(company) {
  const data = JSON.stringify({
    data: {
      name: company + ' Onboarding',
      workspace: WORKSPACE_GID,
      notes: 'Onboarding project for ' + company
    }
  });

  const options = {
    hostname: 'app.asana.com',
    port: 443,
    path: '/api/1.0/projects',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + ASANA_API_TOKEN,
      'Content-Length': Buffer.byteLength(data)
    }
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        if (res.statusCode === 201) {
          resolve('Project created in Asana');
        } else {
          reject(new Error('Asana API error: ' + res.statusCode + ' - ' + body));
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

const company = process.argv[2] || 'Test Company';

createProject(company)
  .then(result => {
    console.log('✅ ' + result);
    console.log('   Project: ' + company + ' Onboarding');
    process.exit(0);
  })
  .catch(error => {
    console.error('❌ Error: ' + error.message);
    process.exit(1);
  });
```

Save and exit.

### Step 4: Create Slack API Script

Create `slack-api.js`:

```bash
nano slack-api.js
```

Paste the following content:

```javascript
const https = require('https');

const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL;

function postToSlack(message) {
  const urlParts = SLACK_WEBHOOK_URL.replace('https://', '').split('/');
  const hostname = urlParts[0];
  const path = '/' + urlParts.slice(1).join('/');
  
  const data = JSON.stringify({ text: message });

  const options = {
    hostname: hostname,
    port: 443,
    path: path,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data)
    }
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        if (res.statusCode === 200) {
          resolve('Message posted to Slack');
        } else {
          reject(new Error('Slack API error: ' + res.statusCode + ' - ' + body));
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

const company = process.argv[2] || 'Test Company';
const dealValue = process.argv[3] || '50000';

const message = '🎉 Great news! We closed the ' + company + ' deal for $' + dealValue + '!';

postToSlack(message)
  .then(result => {
    console.log('✅ ' + result);
    console.log('   Message: ' + message);
    process.exit(0);
  })
  .catch(error => {
    console.error('❌ Error: ' + error.message);
    process.exit(1);
  });
```

Save and exit.

### Step 5: Create Master Orchestration Script

Create `process-deal.js`:

```bash
nano process-deal.js
```

Paste the following content:

```javascript
const { execSync } = require('child_process');

const company = process.argv[2] || 'Globex Corporation';
const dealValue = process.argv[3] || '50000';

console.log('🚀 Processing deal for ' + company + ' ($' + dealValue + ')...\n');

try {
  console.log('📊 Creating Monday.com deal...');
  execSync('node monday-api.js "' + company + '" ' + dealValue + ' "Closed Win"', { stdio: 'inherit' });
  
  console.log('\n📋 Creating Asana project...');
  execSync('node asana-api.js "' + company + '"', { stdio: 'inherit' });
  
  console.log('\n💬 Posting to Slack...');
  execSync('node slack-api.js "' + company + '" ' + dealValue, { stdio: 'inherit' });
  
  console.log('\n✅ All done! Deal processed successfully.');
  
} catch (error) {
  console.error('\n❌ Error processing deal:', error.message);
  process.exit(1);
}
```

Save and exit.

### Step 6: Create Bash Wrapper Script

Create the wrapper script in the home directory:

```bash
cd /home/ubuntu
nano moltbot-deal.sh
```

Paste the following content (replace the tokens with your actual API credentials):

```bash
#!/bin/bash
# Moltbot Deal Processing Script
# Usage: ./moltbot-deal.sh "Company Name" 50000

COMPANY="$1"
VALUE="$2"

cd /home/ubuntu/moltbot-actions

export MONDAY_API_TOKEN="YOUR_MONDAY_API_TOKEN_HERE"
export ASANA_API_TOKEN="YOUR_ASANA_API_TOKEN_HERE"
export SLACK_WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL_HERE"

node process-deal.js "$COMPANY" "$VALUE"
```

Save and exit, then make it executable:

```bash
chmod +x /home/ubuntu/moltbot-deal.sh
```

### Step 7: Test the Automation Manually

Test that the automation works:

```bash
/home/ubuntu/moltbot-deal.sh "Test Company" 50000
```

You should see output indicating that deals were created in Monday.com, Asana, and Slack.

## Part 3: Create the Monitor Script

### Step 1: Get the Session File Path

Start OpenClaw to create a session file:

```bash
openclaw gateway
```

Wait for it to start, then send yourself a test WhatsApp message. Then stop OpenClaw with `Ctrl+C`.

Find the session file:

```bash
ls -la ~/.openclaw/agents/main/sessions/*.jsonl
```

Note the filename (it will be a UUID like `a494b4cb-7c77-44d2-960c-7e314a4eddd4.jsonl`).

### Step 2: Create the Monitor Script

Create the monitor script:

```bash
cd /home/ubuntu
nano monitor-deals.js
```

Paste the following content (replace `SESSION_FILE_NAME` with your actual session file name):

```javascript
const fs = require('fs');
const { execSync } = require('child_process');

const SESSION_FILE = '/home/ubuntu/.openclaw/agents/main/sessions/SESSION_FILE_NAME.jsonl';
let lastPosition = 0;

if (fs.existsSync(SESSION_FILE)) {
  lastPosition = fs.statSync(SESSION_FILE).size;
  console.log('📊 Starting monitor at position:', lastPosition);
}

function extractDealInfo(text) {
  const pattern1 = /(?:closed|won)\s+(?:the\s+)?([\w\s&.,'-]+?)\s+deal\s+for\s+\$?([\d,]+(?:\.\d+)?)[kKmM]?/i;
  const pattern2 = /closed\s+([\w\s&.,'-]+?)\s+for\s+\$?([\d,]+(?:\.\d+)?)[kKmM]?/i;
  
  let match = text.match(pattern1) || text.match(pattern2);
  
  if (!match) return null;
  
  let company = match[1].trim();
  let amountStr = match[2].replace(/,/g, '');
  let amount = parseFloat(amountStr);
  
  if (/\d+[kK]/.test(text)) {
    amount = amount * 1000;
  } else if (/\d+[mM]/.test(text)) {
    amount = amount * 1000000;
  }
  
  return { company, amount: Math.round(amount) };
}

function processNewMessages() {
  if (!fs.existsSync(SESSION_FILE)) return;
  
  const currentSize = fs.statSync(SESSION_FILE).size;
  
  if (currentSize > lastPosition) {
    const fd = fs.openSync(SESSION_FILE, 'r');
    const buffer = Buffer.alloc(currentSize - lastPosition);
    fs.readSync(fd, buffer, 0, buffer.length, lastPosition);
    fs.closeSync(fd);
    
    const newContent = buffer.toString('utf8');
    const lines = newContent.split('\n').filter(line => line.trim());
    
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        
        if (entry.type === 'message' && entry.message?.role === 'user') {
          const content = entry.message.content;
          if (Array.isArray(content)) {
            for (const item of content) {
              if (item.type === 'text') {
                const dealInfo = extractDealInfo(item.text);
                if (dealInfo) {
                  console.log('🎯 Deal detected:', dealInfo);
                  console.log('🚀 Executing automation...');
                  
                  try {
                    execSync(
                      `/home/ubuntu/moltbot-deal.sh "${dealInfo.company}" ${dealInfo.amount}`,
                      { encoding: 'utf8', stdio: 'inherit' }
                    );
                    console.log('✅ Automation completed');
                  } catch (error) {
                    console.error('❌ Error:', error.message);
                  }
                }
              }
            }
          }
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
    
    lastPosition = currentSize;
  }
}

console.log('👀 Monitoring for deal messages...');
setInterval(processNewMessages, 2000);
```

Save and exit.

## Part 4: Set Up PM2 for Automatic Monitoring

### Step 1: Install PM2

```bash
sudo npm install -g pm2
```

### Step 2: Start the Monitor with PM2

```bash
pm2 start /home/ubuntu/monitor-deals.js --name deal-monitor
```

### Step 3: Configure Auto-Start on Reboot

```bash
pm2 startup
```

This will output a command starting with `sudo env PATH=...`. Copy and run that entire command.

Then save the PM2 configuration:

```bash
pm2 save
```

### Step 4: Verify PM2 is Running

```bash
pm2 status
```

You should see `deal-monitor` with status `online`.

## Part 5: Final Testing

### Step 1: Start OpenClaw

In one terminal:

```bash
openclaw gateway
```

Wait for "Listening for personal WhatsApp inbound messages".

### Step 2: Send a Test Message

Send a WhatsApp message to your configured number:

```
We just closed the Saroy Group Inc deal for $50k
```

### Step 3: Verify the Results

Check that:
- Monday.com has a new deal for "Saroy Group Inc" with value $50,000
- Asana has a new project called "Saroy Group Inc Onboarding"
- Slack has a celebration message about the deal

## Troubleshooting

### Check PM2 Logs

```bash
pm2 logs deal-monitor
```

### Check OpenClaw Logs

```bash
tail -f /tmp/openclaw/openclaw-*.log
```

### Restart the Monitor

```bash
pm2 restart deal-monitor
```

### Test the Automation Manually

```bash
/home/ubuntu/moltbot-deal.sh "Test Company" 75000
```

## System Maintenance

The system is now fully automated. The only manual step required is starting OpenClaw:

```bash
ssh ubuntu@YOUR_VM_IP
openclaw gateway
```

The PM2 monitor will run automatically in the background and restart on system reboots.

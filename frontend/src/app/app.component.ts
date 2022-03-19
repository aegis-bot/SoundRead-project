import { Component } from '@angular/core';
import { HttpHandlingService } from './http-handling.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'frontend';
  public fileName: string;
  public file: File;

  public resultFile: File;
  public result: any;

  public sent: boolean = false;
  
  constructor(private dataService: HttpHandlingService){  }
  changeUploadedFile(file:File) {
    console.log("file added to app.component");
    this.file = file;
  }

 

  configureSendButton(cond: boolean) {
    if(cond) {
      (document.getElementById("sendBtn") as any).disabled = false;

    }

    else {
      (document.getElementById("sendBtn") as any).disabled = true;
    }
  }


  sendFilesForTranscribing() {
    //console.log("sendFilesForTranscribing");
    //this.dataService.sendFiles(this.file);
    //REMOVE THIS ONCE BACKEND IS COMPLETED
    this.resultFile = this.file;
    this.sent = true;
  }
}


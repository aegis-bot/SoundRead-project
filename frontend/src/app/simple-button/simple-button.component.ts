import { Injectable, Input, Output, EventEmitter } from '@angular/core';
import { HttpClient, HttpHandler, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

import { Component, OnInit } from '@angular/core';
import { HttpHandlingService } from '../http-handling.service';

@Component({
  selector: 'app-simple-button',
  templateUrl: './simple-button.component.html',
  styleUrls: ['./simple-button.component.css']
})
export class SimpleButtonComponent implements OnInit {
  
  @Input() displayText = "";
  @Input() buttonClass= "btn btn-primary btn-lg";
  @Input() isEnabled: boolean = true;
  constructor() { }
  ngOnInit(): void {
  }

  /*
  public displayText = "Click to Transcribe!";
  public  fileName: any;
  private file: File;
  constructor(private dataService: HttpHandlingService) { }
  ngOnInit(): void {
  }

  
  public onFileSelected($event: any) {    
    this.file = (<HTMLInputElement>event.target).files[0];
    if(this.file) {
      this.fileName = this.file.name
    }
  }

  sendFilesForTranscribing() {
    console.log("sendFilesForTranscribing");
    let input = "I'm screwed"
    this.dataService.sendFiles(this.file);
    //this.dataService.simpleMessageTest();
  
    // send packet to a backend python server

  }
  */
  
}
